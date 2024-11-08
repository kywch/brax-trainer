import time
import functools

import brax.envs
import pufferlib
import pufferlib.vector

from brax_trainer.environment import BraxPufferWrapper

DEFAULT_TIMEOUT = 10


def env_creator(num_envs=128, env_name="ant", episode_length=1000):
    env = brax.envs.create(
        env_name, batch_size=num_envs, episode_length=episode_length, backend="spring"
    )

    env = BraxPufferWrapper(env)
    return env


def test_puffer_env():
    vecenv = pufferlib.vector.make(
        env_creator,
        backend=pufferlib.vector.Native,
    )

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


def profile_env_sps(num_envs, timeout=DEFAULT_TIMEOUT):
    vecenv = pufferlib.vector.make(
        functools.partial(env_creator, num_envs=num_envs),
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

    print(f"num_envs: {num_envs}, SPS: {sps:.1f}")
    return sps


if __name__ == "__main__":
    test_puffer_env()

    for num_envs in [1, 8, 64, 256, 1024, 2048, 4096, 8192]:
        profile_env_sps(num_envs)
