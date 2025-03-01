from typing import Callable, Tuple
from gymnasium import spaces
import torch as th
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
    """
    Custom actor-critic network for PPO.
    Implements separate policy and value networks.
    """

    def __init__(self, feature_dim: int, action_dim: int):
        super(CustomNetwork, self).__init__()

        # Actor network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, action_dim),
        )

        # Critic network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

        # Store latent sizes (used internally in SB3)
        self.latent_dim_pi = 128  # Last layer before action output
        self.latent_dim_vf = 128  # Last layer before value output

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass for both actor and critic.
        :param features: Input features (from SB3's feature extractor).
        :return: (policy output, value output)
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net[:-1](features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net[:-1](features) 


class CustomActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic Policy for PPO.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable SB3's default orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """
        Replaces SB3's default MLP with our custom architecture.
        """
        action_dim = self.action_space.n if isinstance(self.action_space, spaces.Discrete) else self.action_space.shape[0]
        self.mlp_extractor = CustomNetwork(self.features_dim, action_dim)
