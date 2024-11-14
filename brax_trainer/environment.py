from collections import deque

import jax
import numpy as np
from gymnasium import spaces

import brax.envs
from brax.envs.base import PipelineEnv
from brax.io import image

from pufferlib.environment import PufferEnv

BRAX_ENVS = list(brax.envs._envs.keys())


# CHECK ME: PufferEnv already implements vecenv API
def make_vecenv(env_name, args_dict, **env_kwargs) -> PufferEnv:
    assert env_name in BRAX_ENVS, f"Invalid env_name: {env_name}"

    brax_kwargs = {
        "env_name": env_name,
        "batch_size": args_dict["train"]["num_envs"],
        "backend": "spring",
    }

    env_kwargs = args_dict["env"]
    env_kwargs.update(
        {
            "seed": args_dict["train"]["seed"],
            "device": args_dict["train"]["device"],
        }
    )

    # TODO: make brax envs deterministic when torch_deterministic is True

    env = brax.envs.create(**brax_kwargs)
    env = BraxPufferWrapper(env, **env_kwargs)
    return env


class BraxPufferWrapper(PufferEnv):
    """A wrapper that converts batched Brax Env to one that follows PufferEnv API.
    Also implements the Pufferlib EpisodeStats wrapper.
    """

    def __init__(
        self,
        env: PipelineEnv,
        buf=None,
        seed: int = 0,
        reward_scaling: float = 0.1,
        device: str = None,
        **kwargs,
    ):
        self._env = env
        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        if seed is not None:
            self.seed(seed)
        self.device = device
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

        self._reset = jax.jit(reset, backend=self.device)

        def step(state, action):
            state = self._env.step(state, action)
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.device)

        # PufferEnv API check
        super().__init__(buf)

        # reward-related
        self.reward_scaling = reward_scaling
        self.cumulative_reward = np.zeros(self.num_agents, dtype=np.float32)

        # buffer for episode stats
        self.done_envs = np.ones(self.num_agents, dtype=bool)
        self.info_steps = np.zeros(self.num_agents, dtype=np.int32)
        self.finished_episodes = 0
        # NOTE: keeping the last num_envs episode. Revisit this later
        self.episode_returns = deque(maxlen=self.num_agents)
        self.episode_lengths = deque(maxlen=self.num_agents)

    def reset(self, seed=None, options=None):
        """Resets all environments."""
        if seed is not None:
            self.seed(seed)

        self._state, obs, self._key = self._reset(self._key)

        self.observations[:] = jax.device_get(obs)

        # Reset the buffers
        self.masks[:] = True
        self.done_envs[:] = False
        self.cumulative_reward[:] = 0
        self.finished_episodes = 0
        self.episode_returns.clear()
        self.episode_lengths.clear()

        # NOTE: Puffer VecEnvs must return info as a list of dicts
        return self.observations, [{}]

    def step(self, action):
        self._state, obs, reward, done, info = self._step(self._state, action)

        self.observations[:] = jax.device_get(obs)
        self.rewards[:] = jax.device_get(reward)
        self.terminals[:] = jax.device_get(done)
        self.truncations[:] = jax.device_get(info["truncation"])
        self.info_steps[:] = jax.device_get(info["steps"])

        # Check done episodes
        # self.masks[:] = ~(self.terminals | self.truncations)
        np.logical_or(self.terminals, self.truncations, out=self.done_envs)
        np.logical_not(self.done_envs, out=self.masks)

        # CHECK ME: where to put reward shaping?

        # Process rewards
        self.cumulative_reward += self.rewards
        np.multiply(self.rewards, self.reward_scaling, out=self.rewards)

        # Process finished episodes
        if self.done_envs.sum() > 0:
            self.finished_episodes += self.done_envs.sum()
            self.episode_lengths.extend(self.info_steps[self.done_envs])
            self.episode_returns.extend(self.cumulative_reward[self.done_envs])
            self.cumulative_reward[self.done_envs] = 0

        new_info = {
            "finished_episodes": self.finished_episodes,
            "episode_returns": np.mean(self.episode_returns),
            "episode_lengths": np.mean(self.episode_lengths),
        }

        # TODO: info has a lot of info, including reward-related. Report back some of them.

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
