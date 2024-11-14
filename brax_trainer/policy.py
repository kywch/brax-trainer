import numpy as np

import torch
import torch.nn as nn

import pufferlib
import pufferlib.frameworks.cleanrl
from pufferlib.pytorch import layer_init


# This replaces gymnasium's NormalizeObservation wrapper
# NOTE: Tried BatchNorm1d with momentum=None, but the policy did not learn. Check again later.
# CHECK ME: To normalize obs, dividing by a constant is good, but each mujoco/brax env has a different scale...
class RunningNorm(nn.Module):
    def __init__(self, shape: int, epsilon=1e-5, clip=10.0):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros((1, shape), dtype=torch.float32))
        self.register_buffer("running_var", torch.ones((1, shape), dtype=torch.float32))
        self.register_buffer("count", torch.ones(1, dtype=torch.float32))
        self.epsilon = epsilon
        self.clip = clip

    def forward(self, x):
        return torch.clamp(
            (x - self.running_mean.expand_as(x))
            / torch.sqrt(self.running_var.expand_as(x) + self.epsilon),
            -self.clip,
            self.clip,
        )

    @torch.jit.ignore
    def update(self, x):
        # NOTE: Separated update from forward to compile the policy
        # update() must be called to update the running mean and var
        if self.training:
            with torch.no_grad():
                x = x.float()
                assert x.dim() == 2, "x must be 2D"
                mean = x.mean(0, keepdim=True)
                var = x.var(0, unbiased=False, keepdim=True)
                weight = 1 / self.count
                self.running_mean = self.running_mean * (1 - weight) + mean * weight
                self.running_var = self.running_var * (1 - weight) + var * weight
                self.count += 1

    # NOTE: below are needed to torch.save() the model
    @torch.jit.ignore
    def __getstate__(self):
        return {
            "running_mean": self.running_mean,
            "running_var": self.running_var,
            "count": self.count,
            "epsilon": self.epsilon,
            "clip": self.clip,
        }

    @torch.jit.ignore
    def __setstate__(self, state):
        self.running_mean = state["running_mean"]
        self.running_var = state["running_var"]
        self.count = state["count"]
        self.epsilon = state["epsilon"]
        self.clip = state["clip"]


# TODO: test 128 width, with the lstm
class CleanRLPolicy(pufferlib.frameworks.cleanrl.Policy):
    def __init__(self, envs, hidden_size=64):
        super().__init__(policy=None)  # Just to get the right init
        self.is_continuous = True

        self.obs_size = np.array(envs.single_observation_space.shape).prod()
        action_size = np.prod(envs.single_action_space.shape)

        self.obs_norm = torch.jit.script(RunningNorm(self.obs_size))

        # Learn to walk in 20 min: https://arxiv.org/abs/2208.07860
        # Used LayerNorm to regularize the critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.actor_encoder = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor_decoder_mean = layer_init(nn.Linear(hidden_size, action_size), std=0.01)
        self.actor_decoder_logstd = nn.Parameter(torch.zeros(1, action_size))

    def get_value(self, x):
        x = x.float()
        x = self.obs_norm(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.float()
        x = self.obs_norm(x)
        batch = x.shape[0]

        encoding = self.actor_encoder(x)
        action_mean = self.actor_decoder_mean(encoding)
        action_logstd = self.actor_decoder_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)

        # Tried LeanRL, but got error.
        # NOTE: Work around for CUDA graph capture. This produces nans, so disabling for now
        # if torch.cuda.is_current_stream_capturing():
        #     if action is None:
        #         action = action_mean + action_std * torch.randn_like(action_mean)

        #     # Avoid using the torch.distributions.Normal
        #     log_probs = (-0.5 * (((action - action_mean) / action_std) ** 2 + 2 * action_std.log() + torch.log(torch.tensor(2 * torch.pi)))).sum(1)
        #     logits_entropy = (action_std.log() + 0.5 * torch.log(torch.tensor(2 * torch.pi * torch.e))).sum(1)

        # else:

        logits = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = logits.sample()
        log_probs = logits.log_prob(action.view(batch, -1)).sum(1)
        logits_entropy = logits.entropy().sum(1)

        # NOTE: entropy can go negative, when std is small (e.g. 0.1)
        return action, log_probs, logits_entropy, self.critic(x)

    def update_obs_stats(self, x):
        self.obs_norm.update(x)
