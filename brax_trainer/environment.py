"""Wrappers to convert brax envs to pufferlib envs."""

from typing import Optional

import jax
import numpy as np

from gymnasium import spaces
from brax.envs.base import PipelineEnv
from brax.io import image

from pufferlib.environment import PufferEnv


class BraxPufferWrapper(PufferEnv):
    """A wrapper that converts batched Brax Env to one that follows PufferEnv API.
    Also implements the Pufferlib EpisodeStats wrapper.
    """

    def __init__(self, env: PipelineEnv, buf=None, seed: int = 0, backend: Optional[str] = None):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.seed(seed)
        self.backend = backend
        self._state = None

        # Since there is only one agent per env, treat num_agents as num_envs
        self.num_envs = 1
        self.num_agents = self._env.batch_size

        obs = np.inf * np.ones(self._env.observation_size, dtype="float32")
        self.single_observation_space = spaces.Box(-obs, obs, dtype="float32")

        action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        self.single_action_space = spaces.Box(action[:, 0], action[:, 1], dtype="float32")

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

        # PufferEnv API check
        super().__init__(buf)

        # buffer for episode stats
        self.episode_length = np.zeros(self.num_agents, dtype=np.float32)
        self.episode_return = np.zeros(self.num_agents, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self._state, obs, self._key = self._reset(self._key)

        self.observations[:] = jax.device_get(obs)

        # NOTE: Puffer VecEnvs must return info as a list of dicts
        return self.observations, [{}]

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)

        self.observations[:] = jax.device_get(obs)
        self.rewards[:] = jax.device_get(reward)
        self.terminals[:] = jax.device_get(done)
        self.truncations[:] = jax.device_get(info["truncation"])

        # Process episode stats for done episodes
        self.masks[:] = ~(self.terminals | self.truncations)
        self.episode_length[:] = jax.device_get(info["steps"])

        done_envs = ~self.masks
        new_info = (
            {}
            if done_envs.sum() == 0
            else dict(
                episode_return=np.mean(self.rewards[done_envs]),
                episode_length=np.mean(self.episode_length[done_envs]),
            )
        )

        # TODO: info has a lot of info. Report back some of them.

        return self.observations, self.rewards, self.terminals, self.truncations, [new_info]

    def close(self):
        # NOTE: Brax envs don't require explicit cleanup
        pass

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode="human"):
        if mode == "rgb_array":
            sys, state = self._env.sys, self._state
            if state is None:
                raise RuntimeError("must call reset or step before rendering")
            return image.render_array(sys, state.pipeline_state.take(0), 256, 256)
        else:
            return super().render(mode=mode)  # just raise an exception
