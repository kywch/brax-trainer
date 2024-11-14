import time

from brax_trainer.environment import make_vecenv

DEFAULT_TIMEOUT = 10


def init_wandb(args_dict, run_name, id=None, resume=True, disable=False):
    import wandb

    if disable is True:
        wandb.init(mode="disabled")
    else:
        wandb.init(
            id=id or wandb.util.generate_id(),
            project=args_dict["wandb_project"],
            group=args_dict["wandb_group"],
            allow_val_change=True,
            save_code=True,
            resume=resume,
            config=args_dict,
            name=run_name,
        )

    return wandb


def profile_env_sps(env_name, args_dict, timeout=DEFAULT_TIMEOUT):
    vecenv = make_vecenv(env_name, args_dict)
    num_envs = args_dict["train"]["num_envs"]

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

    print(f"num_envs: {num_envs}, SPS: {sps:.1f}")
    return sps
