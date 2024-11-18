import argparse
import tomllib
import signal
import random
import uuid
import time
import ast
import os

import torch
import numpy as np

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from PIL import Image
from tqdm import tqdm
from rich.traceback import install

import brax_trainer.clean_pufferl as clean_pufferl
import brax_trainer.environment as environment
import brax_trainer.policy as policy_module
from brax_trainer.utils import init_wandb, add_text_to_image, create_video

# Rich tracebacks
install(show_locals=False)

# Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def parse_args(config="config/debug.toml"):
    parser = argparse.ArgumentParser(description="Training arguments for brax", add_help=False)
    parser.add_argument("-c", "--config", default=config)
    parser.add_argument(
        "-e",
        "--env-name",
        type=str,
        default="ant",
        choices=environment.BRAX_ENVS,
        help="Name of specific environment to run",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="sweep",
        choices="train sweep eval video check_sps cprofile".split(),
    )

    # parser.add_argument("--eval-model-path", type=str, default=None)
    parser.add_argument(
        "-p",
        "--eval-model-path",
        type=str,
        # default=None,
        default="experiments/ant-48a31ec9/model_000046.pt",
    )

    # parser.add_argument(
    #     "--baseline", action="store_true", help="Pretrained baseline where available"
    # )

    parser.add_argument(
        "--exp-id", "--exp-name", type=str, default=None, help="Resume from experiment"
    )
    parser.add_argument("--wandb-project", type=str, default="brax")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--track", action="store_true", help="Track on WandB")
    parser.add_argument(
        "--repeat", type=int, default=1, help="Repeat the training with different seeds"
    )
    parser.add_argument("-d", "--device", type=str, default=None)
    parser.add_argument("-s", "--seed", type=int, default=None)

    args = parser.parse_known_args()[0]

    # Load config file
    if not os.path.exists(args.config):
        raise Exception(f"Config file {args.config} not found")
    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    for section in config:
        for key in config[section]:
            argparse_key = f"--{section}.{key}".replace("_", "-")
            parser.add_argument(argparse_key, default=config[section][key])

    # Override config with command line arguments
    parsed = parser.parse_args().__dict__

    args = {"env": {}, "policy": {}, "rnn": {}}
    env_name = parsed.pop("env_name")
    for key, value in parsed.items():
        next = args
        for subkey in key.split("."):
            if subkey not in next:
                next[subkey] = {}
            prev = next
            next = next[subkey]
        try:
            prev[subkey] = ast.literal_eval(value)
        except:  # noqa
            prev[subkey] = value

    # Determine device with priority order
    # Priority: args["device"] > args["train"]["device"] > torch.cuda.is_available()
    device = None
    if args.get("device") is not None:
        device = args["device"]
    elif args["train"].get("device") is not None:
        device = args["train"]["device"]
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fallback to CPU if CUDA is requested but not available
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    # Update both locations with the determined device
    args["train"]["device"] = args["device"] = device

    return args, env_name


def args_to_session_spec(args):
    if args["seed"] is not None:
        args["train"]["seed"] = args["seed"]

    args["env"].update(
        {
            "seed": args["train"]["seed"],
            "device": args["device"],
        }
    )

    return {
        "seed": args["train"]["seed"],
        "device": args["device"],
        "env_config": {
            "brax_kwargs": {
                "env_name": env_name,
                "batch_size": args["train"]["num_envs"],
            },
            "env_kwargs": args["env"],
        },
        "policy_config": {
            "policy_cls_name": args["base"]["policy_name"],
            "policy_kwargs": args["policy"],
            "rnn_cls_name": args["base"].get("rnn_name", None),
            "rnn_kwargs": args["rnn"],
        },
        "train_config": args["train"],
        "state_dict": None,
    }


def make_env_and_policy(session_spec):
    vecenv = pufferlib.vector.make(
        environment.make_vecenv,
        env_kwargs=session_spec["env_config"],
        backend=pufferlib.vector.Native,
    )

    # Env compile & warm up
    print("Warming up the jax environment...")
    vecenv.reset()
    actions = vecenv.action_space.sample()
    vecenv.step(actions)

    # Make policy
    policy_config = session_spec["policy_config"]
    policy_cls = getattr(policy_module, policy_config["policy_cls_name"])
    rnn_cls = None
    if policy_config["rnn_cls_name"] is not None:
        rnn_cls = getattr(policy_module, policy_config["rnn_cls_name"])

    policy = policy_cls(vecenv.driver_env, **policy_config["policy_kwargs"])
    if isinstance(policy, pufferlib.frameworks.cleanrl.Policy) or isinstance(
        policy, pufferlib.frameworks.cleanrl.RecurrentPolicy
    ):
        pass
    elif rnn_cls is not None:
        policy = rnn_cls(vecenv.driver_env, policy, **policy_config["rnn_kwargs"])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    if session_spec["state_dict"] is not None:
        policy.load_state_dict(session_spec["state_dict"])

    policy.to(session_spec["device"])

    return vecenv, policy


def train(args, wandb=None, skip_dash=False):
    session_spec = args_to_session_spec(args)
    vecenv, policy = make_env_and_policy(session_spec)

    train_config = pufferlib.namespace(
        **session_spec["train_config"],
        env=env_name,
        exp_id=args["exp_id"] or env_name + "-" + str(uuid.uuid4())[:8],
    )
    data = clean_pufferl.create(
        train_config, vecenv, policy, session_spec, wandb=wandb, skip_dash=skip_dash
    )

    try:
        while data.global_step < train_config.total_timesteps:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)

        uptime = data.profile.uptime

        returns = []
        if "sweep" in args:
            # TODO: If we CARBS reward param, we should have a separate venenv ready for eval
            # Run evaluation to get the average stats

            target_metric = args["sweep"]["metric"]["name"].split("/")[-1]
            data.vecenv.reset(seed=args["seed"])
            num_eval_epochs = train_config.eval_timesteps // train_config.batch_size
            for _ in range(1 + num_eval_epochs):
                _, infos = clean_pufferl.evaluate(data)
                returns.extend(infos[target_metric])

    except Exception as e:  # noqa
        uptime, returns = 0, []
        import traceback

        traceback.print_exc()

    clean_pufferl.close(data)
    return returns, uptime


def run_sweep(args, env_name):
    import wandb
    from brax_trainer.carbs_sweep import init_carbs, carbs_runner_fn

    # if not os.path.exists("carbs_checkpoints"):
    #     os.system("mkdir carbs_checkpoints")

    carbs = init_carbs(args, num_random_samples=10)

    sweep_id = wandb.sweep(
        sweep=args["sweep"],
        project="carbs",
    )

    def train_fn(args, wandb):
        return train(args, wandb=wandb, skip_dash=True)

    # Run sweep
    wandb.agent(
        sweep_id,
        carbs_runner_fn(args, env_name, carbs, sweep_id, train_fn),
        count=args["train"]["num_sweeps"],
    )


def evaluate(args, num_envs=2048, rollout_steps=1001):
    # Load the pre-trained model
    model_path = args["eval_model_path"]
    assert model_path is not None, "model_path must be provided for record_video"

    session_spec = torch.load(model_path)
    assert session_spec["state_dict"] is not None, "model_path must contain a state_dict"

    # Update the current device
    session_spec["device"] = args["device"]

    # Make the single env for the video
    session_spec["env_config"]["brax_kwargs"]["batch_size"] = num_envs

    vecenv, policy = make_env_and_policy(session_spec)
    policy.eval()

    # Update the eval_timesteps based on the num_envs and rollout_steps
    session_spec["train_config"]["eval_timesteps"] = num_envs * rollout_steps
    train_config = pufferlib.namespace(
        **session_spec["train_config"],
        env=env_name,
        exp_id=args["exp_id"] or env_name + "-" + str(uuid.uuid4())[:8],
    )
    data = clean_pufferl.create(train_config, vecenv, policy, session_spec, skip_dash=True)

    returns = []
    lengths = []
    data.vecenv.reset(seed=args["seed"])
    num_eval_epochs = train_config.eval_timesteps // train_config.batch_size
    for _ in range(1 + num_eval_epochs):
        _, infos = clean_pufferl.evaluate(data)
        returns.extend(infos["episode_return"])
        lengths.extend(infos["episode_length"])

    print(f"\n\nEvaluation results with {num_envs} envs * {rollout_steps} steps")
    print(f"Episode count: {len(returns)}")
    print(f"Episode lengths: {np.mean(lengths):.2f}")
    print(f"Episode returns: {np.mean(returns):.2f}")

    clean_pufferl.close(data)


def record_video(args, rollout_steps=2000):
    # Load the pre-trained model
    model_path = args["eval_model_path"]
    assert model_path is not None, "model_path must be provided for record_video"

    session_spec = torch.load(model_path)
    assert session_spec["state_dict"] is not None, "model_path must contain a state_dict"

    # Update the current device
    session_spec["device"] = args["device"]

    # Make the single env for the video
    session_spec["env_config"]["brax_kwargs"]["batch_size"] = 1

    env, policy = make_env_and_policy(session_spec)
    policy.eval()

    # Rollout
    frames = []
    episode = 1
    ep_reward = 0

    obs, _ = env.reset(seed=args["seed"])
    obs_list = [obs.squeeze()]
    state = None

    for tick in tqdm(range(rollout_steps)):
        with torch.no_grad():
            obs = torch.from_numpy(obs).to(args["device"]).view(1, -1)
            if hasattr(policy, "lstm"):
                action, _, _, _, state = policy(obs, state)
            else:
                action, _, _, _ = policy(obs)

            action = action.cpu().numpy().reshape(env.action_space.shape)

        obs, reward, done, truncated, infos = env.step(action)
        obs_list.append(obs)

        rgb_array = env.render()

        if done or truncated:
            print(f"Tick: {tick}, Episode: {episode}, Reward: {ep_reward:.4f}")
            episode += 1
            # Reset is handled by the env
        else:
            ep_reward = env.cumulative_reward[0]

        # Add episode, reward and tick to the image
        image = Image.fromarray(rgb_array)
        image = add_text_to_image(
            image,
            f"Tick {tick+1}/{rollout_steps}\nEpisode: {episode}, Reward: {ep_reward:.4f}",
            (10, 10),
        )
        frames.append(np.array(image))

        # print(f"Reward: {reward:.4f}, Tick: {tick}, Done: {done}")
        # print(f"Next action: {action[0]}")

    # Save the video file to the model path
    video_name = f"{model_path.split('.pt')[0]}_video.mp4"
    create_video(frames, video_name, fps=30)

    # Get some basic stats on obs, to see if it needs some preprocessing
    obs_mat = np.vstack(obs_list)
    print(f"Max abs col mean: {abs(obs_mat.mean(axis=1)).max()}")
    print(f"Max abs col std: {obs_mat.std(axis=1).max()}")
    print(f"Min and max obs: {obs_mat.min()}, {obs_mat.max()}")


def profile_env_sps(env_config, timeout=10):
    vecenv = pufferlib.vector.make(
        environment.make_vecenv,
        env_kwargs=env_config,
        backend=pufferlib.vector.Native,
    )

    actions = [vecenv.action_space.sample() for _ in range(1000)]

    # warmup
    vecenv.reset()
    vecenv.step(actions[0])

    agent_steps = 0
    vecenv.async_reset()

    # profile
    start = time.time()
    while time.time() - start < timeout:
        vecenv.send(actions[agent_steps % 1000])
        o, r, d, t, i, env_id, mask = vecenv.recv()
        agent_steps += sum(mask)

    sps = agent_steps / (time.time() - start)
    vecenv.close()

    print(f"num_envs: {env_config['brax_kwargs']['batch_size']}, SPS: {sps:.1f}")
    return sps


if __name__ == "__main__":
    args, env_name = parse_args()

    # Process mode
    if args["mode"] == "train":
        assert args["repeat"] > 0, "Repeat count must be positive"
        if args["repeat"] > 1:
            args["track"] = True
            assert args["wandb_group"] is not None, "Repeating requires a wandb group"
        wandb = None

        for i in range(args["repeat"]):
            if i > 0:
                # Generate a new 8-digit seed
                args["train"]["seed"] = random.randint(10_000_000, 99_999_999)

            if args["track"]:
                run_name = f"pufferl_{env_name}_{args['train']['seed']}_{int(time.time())}"
                wandb = init_wandb(args, run_name)
            train(args, wandb=wandb)

    elif args["mode"] == "sweep":
        run_sweep(args, env_name)

    elif args["mode"] == "eval":
        evaluate(args)

    elif args["mode"] == "video":
        assert args["eval_model_path"] is not None, "model_path must be provided for video"
        record_video(args)

    elif args["mode"] == "check_sps":
        session_spec = args_to_session_spec(args)
        env_config = session_spec["env_config"]
        for num_envs in [1, 8, 64, 256, 1024, 2048, 4096, 8192, 16384]:
            try:
                env_config["brax_kwargs"]["batch_size"] = num_envs
                profile_env_sps(env_config)
            except Exception as e:
                print(f"Running {num_envs} {env_name} envs failed: {e}")

    elif args["mode"] == "cprofile":
        import cProfile

        session_spec = args_to_session_spec(args)
        session_spec["train_config"]["total_timesteps"] = 1_000_000
        cProfile.run("train(session_spec)", "stats.profile")
        import pstats
        from pstats import SortKey

        p = pstats.Stats("stats.profile")
        p.sort_stats(SortKey.TIME).print_stats(20)
