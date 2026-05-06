import sys
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution

class CustomNetworkMlp(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        self.features_in_dim = feature_dim;

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # self.device = "cuda:0"

        # 纯 MLP 策略：与 MlpMpcPolicy 对照使用
        # 这里 policy/value 都是标准前馈网络，不包含 MPC 求解环节
        # latent_dim_* 会被 SB3 的分布头/价值头继续消费

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        # 与 MPC 版本不同：这里 actor 直接输出潜变量，不依赖环境状态的动力学结构
        features_in = features[:, :self.features_in_dim]
        features_in = th.nan_to_num(features_in, nan=0.0, posinf=0.0, neginf=0.0)
        policy_latent = self.policy_net(features_in)
        return th.nan_to_num(policy_latent, nan=0.0, posinf=1.0, neginf=-1.0)


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        features_in = th.nan_to_num(features_in, nan=0.0, posinf=0.0, neginf=0.0)
        values = self.value_net(features_in)
        return th.nan_to_num(values, nan=0.0, posinf=1e3, neginf=-1e3)


class MlpOnlyPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    

    def _build_mlp_extractor(self) -> None:
        # 注册自定义特征抽取后端：输出 actor/value 两路 latent
        self.mlp_extractor = CustomNetworkMlp(self.features_dim)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        # Guard against NaN/Inf in latent_pi (e.g. from gradient-corrupted weights).
        latent_pi = th.nan_to_num(latent_pi, nan=0.0, posinf=1.0, neginf=-1.0)
        mean_actions = self.action_net(latent_pi)
        mean_actions = th.nan_to_num(mean_actions, nan=0.0, posinf=1.0, neginf=-1.0)
        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        return super()._get_action_dist_from_latent(latent_pi)

