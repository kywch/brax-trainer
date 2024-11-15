import argparse
import tomllib
import signal
import random
import uuid
import time
import ast
import os

import pufferlib
import pufferlib.utils
import pufferlib.vector
import pufferlib.frameworks.cleanrl

from rich.traceback import install

import brax_trainer.clean_pufferl as clean_pufferl
import brax_trainer.environment as environment
import brax_trainer.policy as policy
from brax_trainer.utils import init_wandb

# Rich tracebacks
install(show_locals=False)

# Aggressively exit on ctrl+c
signal.signal(signal.SIGINT, lambda sig, frame: os._exit(0))


def make_policy(env, policy_cls, rnn_cls, args):
    policy = policy_cls(env, **args["policy"])
    if isinstance(policy, pufferlib.frameworks.cleanrl.Policy) or isinstance(
        policy, pufferlib.frameworks.cleanrl.RecurrentPolicy
    ):
        pass
    elif rnn_cls is not None:
        policy = rnn_cls(env, policy, **args["rnn"])
        policy = pufferlib.frameworks.cleanrl.RecurrentPolicy(policy)
    else:
        policy = pufferlib.frameworks.cleanrl.Policy(policy)

    return policy.to(args["train"]["device"])


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
        default="train",
        choices="train video sweep check_sps cprofile".split(),
    )

    # parser.add_argument("--eval-model-path", type=str, default=None)
    parser.add_argument(
        "-p",
        "--eval-model-path",
        type=str,
        default=None,
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

    return args, env_name


def train(args, vecenv_creator, policy_cls, rnn_cls, wandb=None, skip_dash=False):
    # TODO: use puffer.vector.make
    vecenv = vecenv_creator()

    # env compile & warm up
    print("Warming up the jax environment...")
    vecenv.reset()
    actions = vecenv.action_space.sample()
    vecenv.step(actions)

    policy = make_policy(vecenv.driver_env, policy_cls, rnn_cls, args)
    train_config = pufferlib.namespace(
        **args["train"],
        env=env_name,
        exp_id=args["exp_id"] or env_name + "-" + str(uuid.uuid4())[:8],
    )
    data = clean_pufferl.create(train_config, vecenv, policy, wandb=wandb, skip_dash=skip_dash)

    try:
        while data.global_step < train_config.total_timesteps:
            clean_pufferl.evaluate(data)
            clean_pufferl.train(data)

        uptime = data.profile.uptime

        # TODO: If we CARBS reward param, we should have a separate venenv ready for eval
        # Run evaluation to get the average stats
        stats = []
        data.vecenv.async_reset(seed=int(time.time()))
        num_eval_epochs = train_config.eval_timesteps // train_config.batch_size
        for _ in range(1 + num_eval_epochs):  # extra data for sweeps
            stats.append(clean_pufferl.evaluate(data)[0])

    except Exception as e:  # noqa
        uptime, stats = 0, []
        import traceback

        traceback.print_exc()

    clean_pufferl.close(data)
    return stats, uptime


### CARBS Sweeps
def run_sweep(args, env_name, vecenv_creator, policy_cls, rnn_cls):
    import wandb
    from brax_trainer.carbs_sweep import init_carbs, carbs_runner_fn

    if not os.path.exists("carbs_checkpoints"):
        os.system("mkdir carbs_checkpoints")

    carbs = init_carbs(args, num_random_samples=20)

    sweep_id = wandb.sweep(
        sweep=args["sweep"],
        project="carbs",
    )

    def train_fn(args, wandb):
        return train(args, vecenv_creator, policy_cls, rnn_cls, wandb=wandb, skip_dash=True)

    # Run sweep
    wandb.agent(
        sweep_id,
        carbs_runner_fn(args, env_name, carbs, sweep_id, train_fn),
        count=args["train"]["num_sweeps"],
    )


if __name__ == "__main__":
    args, env_name = parse_args()

    if args["device"] is not None:
        args["train"]["device"] = args["device"]

    # Load env binding and policy
    def vecenv_creator():
        return environment.make_vecenv(env_name, args)

    policy_cls = getattr(policy, args["base"]["policy_name"])
    rnn_cls = None
    if "rnn_name" in args["base"]:
        rnn_cls = getattr(policy, args["base"]["rnn_name"])

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
            train(args, vecenv_creator, policy_cls, rnn_cls, wandb=wandb)

    # elif args["mode"] == "video":
    #     # Single env
    #     args["train"]["num_envs"] = 1
    #     args["train"]["num_workers"] = 1
    #     args["train"]["env_batch_size"] = 1

    #     clean_pufferl.rollout(
    #         env_creator[0],
    #         args["env"],
    #         policy_cls=policy_cls,
    #         rnn_cls=rnn_cls,
    #         agent_creator=make_policy,
    #         agent_kwargs=args,
    #         model_path=args["eval_model_path"],
    #         device=args["train"]["device"],
    #     )

    elif args["mode"] == "sweep":
        run_sweep(args, env_name, vecenv_creator, policy_cls, rnn_cls)

    elif args["mode"] == "check_sps":
        from brax_trainer.utils import profile_env_sps

        for num_envs in [1, 8, 64, 256, 1024, 2048, 4096, 8192, 16384]:
            try:
                profile_env_sps(env_name, args_dict={"train": {"num_envs": num_envs}})
            except Exception as e:
                print(f"Running {num_envs} {env_name} envs failed: {e}")

    elif args["mode"] == "cprofile":
        import cProfile

        args["train"]["total_timesteps"] = 1_000_000
        cProfile.run("train(args, vecenv_creator, policy_cls, rnn_cls)", "stats.profile")
        import pstats
        from pstats import SortKey

        p = pstats.Stats("stats.profile")
        p.sort_stats(SortKey.TIME).print_stats(20)
