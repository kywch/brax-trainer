import time

from brax_trainer.environment import make_vecenv
from brax_trainer.utils import profile_env_sps


def test_puffer_env():
    test_kwargs = {"env_name": "ant", "args_dict": {"train": {"num_envs": 1}}}

    vecenv = make_vecenv(**test_kwargs)

    # Sync API
    vecenv.reset(seed=int(time.time()))
    for _ in range(10):
        actions = vecenv.action_space.sample()
        _, _, _, _, _ = vecenv.step(actions)

    # Async API -- clean_pufferl uses this
    vecenv.async_reset()
    for _ in range(10):
        actions = vecenv.action_space.sample()
        _, _, _, _, _, _, _ = vecenv.recv()
        vecenv.send(actions)

    vecenv.close()


if __name__ == "__main__":
    test_puffer_env()

    env_name = "ant"
    for num_envs in [1, 8, 64, 256, 1024, 2048, 4096, 8192]:
        profile_env_sps(env_name, args_dict={"train": {"num_envs": num_envs}})
