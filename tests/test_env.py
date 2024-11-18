import time
import torch

from brax_trainer.environment import make_vecenv


def test_puffer_env():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_config = {
        "brax_kwargs": {"env_name": "ant", "batch_size": 1},
        "env_kwargs": {"seed": 1, "device": device},
    }

    vecenv = make_vecenv(**env_config)

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
