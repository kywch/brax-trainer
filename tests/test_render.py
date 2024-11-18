import time
import torch

from brax_trainer.environment import make_vecenv
from brax_trainer.utils import create_video


def test_brax_render(env_name="ant", video_path="test.mp4"):
    # Single env
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env_config = {
        "brax_kwargs": {"env_name": env_name, "batch_size": 1},
        "env_kwargs": {"seed": 1, "device": device},
    }

    env = make_vecenv(**env_config)

    frames = []
    env.reset(seed=int(time.time()))
    for _ in range(100):
        actions = env.action_space.sample()
        _, _, _, _, _ = env.step(actions)
        frames.append(env.render())

    env.close()

    create_video(frames, video_path)


if __name__ == "__main__":
    test_brax_render()
