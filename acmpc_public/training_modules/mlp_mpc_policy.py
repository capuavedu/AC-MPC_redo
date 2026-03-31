import os

from ..diff_mpc_drones import drone
from ..diff_mpc_drones import il_env

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
    
from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNetwork(nn.Module):
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
        last_layer_dim_pi: int = 4,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        self.features_in_dim = feature_dim;

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        # MPC 时域长度（由环境变量 ACMPC_T 注入）
        self.T = int(os.environ["ACMPC_T"])
        self.n_o = 28
        # 每个时刻输出 28 个参数：
        # - Q 对角元素 14 个（p3 + q4 + v3 + thrust1 + omega3）
        # - p 线性项 14 个（同顺序）
        self.n_output = self.n_o * self.T
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.predictions = th.zeros((self.T, 1, 17)).to(device=self.device)

        # Policy 网络不直接输出动作，而是输出 MPC 代价参数（跨整个预测时域）
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, self.n_output), nn.Sigmoid()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU()
        )


        # mpc_env 负责调用底层可微/可批量 MPC 求解器
        self.mpc_env = il_env.IL_Env("drone", mpc_T=self.T)
        self.dx = drone.DroneDx(device=self.device)
        # warm-start 控制（上一时刻首控制）用于提高求解稳定性/速度
        self.u_prev = None

        print(self.policy_net)
        print(self.value_net)

    def forward(
        self,
        features: th.Tensor,
        states: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """ 
        # SB3 默认仅传 features；在这种情况下用 features 作为状态输入。
        if states is None:
            states = features
        return self.forward_actor(features, states), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor, states: Optional[th.Tensor] = None) -> th.Tensor:

        # SB3 在推理时可能只传 features；此时用 features 作为状态
        if states is None:
            states = features

        states = states.to(self.device).float()
        features_in = features[:, :self.features_in_dim]
        if (states.ndimension() == 1):
            states = th.unsqueeze(states, dim=0)

        # 环境状态只取前 10 维，需与 DroneDx.n_state 一致：[p,q,v]
        states = states[:, 0:10]

        # 前向 MLP：输出 [0,1] 区间代价参数，后续做尺度变换映射到真实范围
        sigmoid_cost_all = self.policy_net(features_in)

        # Solve optimization in smaller batches
        n_batch = features.shape[0]

        chunk_length = 1024
        # n_chunks = n_batch // chunk_length + 1

        chunks = th.split(sigmoid_cost_all, chunk_length, dim=0)
        # 代价缩放参数：
        # - Q 取正值（加 epsilon 防止过小）
        # - p 允许正负（中心化后放缩）
        epsilon = 0.1
        range_Q = 100000.0
        range_p = 100000.0
        range_p_t = 2 * range_Q / 2 * self.dx.mass * 9.806
        n_tau = 14


        # 首次调用无 warm-start 时，使用悬停推力初始化
        if (self.u_prev is None):
            self.u_prev = th.zeros(4, n_batch).to(device=self.device)
            self.u_prev[0, :] = self.dx.mass * 9.806


        # Containers for full solution
        nom_x = th.zeros((n_batch, self.T, self.dx.n_state)).to(device=self.device)
        nom_u = th.zeros((n_batch, self.T, self.dx.n_ctrl)).to(device=self.device)
        idx_start = 0

        for idx, sigmoid_cost in enumerate(chunks):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk
            # 按时域展开的 [Q参数 | p参数]，当前均在 [0,1]
            x_Q = sigmoid_cost[:, :14*self.T].to(device=self.device)
            x_p = sigmoid_cost[:, 14*self.T:].to(device=self.device)

            # Q 对角项分组（均保持正值）
            q_p = x_Q[:, :3*self.T] * range_Q + epsilon
            q_q = x_Q[:, 3*self.T:7*self.T] * range_Q + epsilon
            q_v = x_Q[:, 7*self.T:10*self.T] * range_Q + epsilon
            q_w = x_Q[:, 10*self.T:13*self.T] * range_Q + epsilon
            q_t = x_Q[:, 13*self.T:14*self.T] * range_Q + epsilon

            # p 线性项分组（中心化后可正可负）
            p_p = (x_p[:, :3*self.T] - 0.5) * range_p
            p_q = (x_p[:, 3*self.T:7*self.T] - 0.5) * range_p
            p_v = (x_p[:, 7*self.T:10*self.T] - 0.5) * range_p
            p_w = (x_p[:, 10*self.T:13*self.T] - 0.5) * range_p
            p_t = x_p[:, 13*self.T:14*self.T] * range_p_t + epsilon

            u_prev_chunk = self.u_prev[:, idx_start:idx_end]

            _Q = th.zeros(self.T, n_chunk, n_tau, n_tau, device=self.device)
            _p = th.zeros(self.T, n_chunk, n_tau, device=self.device)


            states_chunk = states[idx_start:idx_end, :]

            for i in range(self.T):

                # 每个时刻构造 14x14 对角代价矩阵 Q_i 和向量 p_i
                Q_diag_embed_i = th.diag_embed(th.cat([q_p[:, i*3:i*3+3],
                                                         q_q[:, i*4:i*4+4],
                                                         q_v[:,i*3:i*3+3],
                                                         q_t[:,i].unsqueeze(1),
                                                         q_w[:,i*3:i*3+3]], dim=1))


                p_i = th.cat([p_p[:, i*3:i*3+3],
                             p_q[:, i*4:i*4+4],
                             p_v[:, i*3:i*3+3],
                             -p_t[:,i].unsqueeze(1),
                             p_w[:,i*3:i*3+3],
                             ], dim=1)


                _Q[i, :,:,:] = Q_diag_embed_i
                _p[i, :, :] = p_i


            # 调用 MPC：输入 (动力学, 初始状态, 全时域 Q/p, warm-start 控制)
            nom_x_chunk, nom_u_chunk = self.mpc_env.mpc(
                self.dx, states_chunk, _Q, _p,
                # u_init=train_warmstart[idxs].transpose(0,1),
                u_init=u_prev_chunk,
                # eps_override=0.1,
                lqr_iter_override=1,
            )

            nom_x[idx_start:idx_end, :, :] = nom_x_chunk.transpose(0,1)
            nom_u[idx_start:idx_end, :, :] = nom_u_chunk.transpose(0,1)
            idx_start = idx_end


        # 更新 warm-start：保留本次最优序列第一步控制作为下次初值
        self.u_prev = nom_u[:,0,:].transpose(0,1)

        self.predictions = th.cat((nom_x, nom_u), dim=2).detach()


        # MPC 输出动作（真实物理量）-> PPO 分布头所需归一化动作
        # 第一维先转为比推力（除以质量）
        thrust = nom_u[:, 0, 0]/self.dx.mass
        # 其余三维是机体系角速度(rad/s)
        omegas = nom_u[:,0,1:4]

        # 归一化约定：环境会在后处理中反归一化
        normalization_max = 8.5 # Max thrust per rotor in Newtons
        force_mean = (normalization_max * 4 / self.dx.mass) / 2.0
        force_std = (normalization_max * 4 / self.dx.mass) / 2.0
        thrust_normalized = (thrust  - force_mean) / force_std

        # print("normalized_thrust_origin")
        # print(thrust_normalized)

        omega_max = th.Tensor([10.0, 10.0, 4.0]).to(device=self.device)

        omegas_normalized = th.div(omegas, omega_max).to(device=self.device)

        inputs_normalized = th.cat((thrust_normalized.unsqueeze(1), omegas_normalized), dim=1).to(self.device)


        return inputs_normalized


    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        features_in = features[:, :self.features_in_dim]
        return self.value_net(features_in)


class MlpMpcPolicy(ActorCriticPolicy):
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
        self.mlp_extractor = CustomNetwork(self.features_dim)

