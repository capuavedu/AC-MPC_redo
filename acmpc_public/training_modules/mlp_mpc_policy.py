import sys
import os

DRONE_PATH = os.path.join(os.path.dirname(__file__), "..", "diff_mpc_drones")


sys.path.append(DRONE_PATH)

import drone
import il_env

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

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


        self.T = int(os.environ["ACMPC_T"])
        self.n_o = 28
        self.n_output = self.n_o * self.T
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.predictions = th.zeros((self.T, 1, 17)).to(device=self.device)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, self.n_output), nn.Sigmoid()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.features_in_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU()
        )


        self.mpc_env = il_env.IL_Env("drone", mpc_T=self.T)
        self.dx = drone.DroneDx(device=self.device)
        self.u_prev = None
        # 周期性内存日志计数与间隔（可通过环境变量调整）
        self._mem_log_counter = 0
        try:
            self._mem_log_interval = int(os.environ.get("MPC_MEM_LOG_INTERVAL", "5000"))
        except Exception:
            self._mem_log_interval = 5000

        print(self.policy_net)
        print(self.value_net)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """ 
        return self.forward_actor(features), self.forward_critic(features)


    def forward_actor(self, features: th.Tensor, states: th.Tensor = None) -> th.Tensor:

        # 周期性资源监控（减少日志频率以免刷屏）
        try:
            self._mem_log_counter += 1
        except Exception:
            self._mem_log_counter = 1

        if self._mem_log_counter % max(1, self._mem_log_interval) == 0:
            try:
                import psutil
                import pynvml
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / 1024 / 1024
                vms_mb = mem_info.vms / 1024 / 1024
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = mem.used / 1024 / 1024
                gpu_mem_total = mem.total / 1024 / 1024
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                print(f"[MPC_POLICY][MEM] RAM: {rss_mb:.1f}MB RSS, {vms_mb:.1f}MB VMS | GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} MB, Util: {gpu_util}%")
            except Exception as e:
                print(f"[MPC_POLICY][MEM] 资源监控失败: {e}")

        if states is None:
            states = features

        states = states.to(self.device).float()
        features_in = features[:, :self.features_in_dim]
        if (states.ndimension() == 1):
            states = th.unsqueeze(states, dim=0)

        # [p, q, v]:
        states = states[:, 0:10]

        # Forward MLP to get cost function for MPC
        sigmoid_cost_all = self.policy_net(features_in)

        # Solve optimization in smaller batches
        n_batch = features.shape[0]

        chunk_length = 1024
        # n_chunks = n_batch // chunk_length + 1

        chunks = th.split(sigmoid_cost_all, chunk_length, dim=0)
        epsilon = 0.1
        range_Q = 100.0
        range_p = 100.0
        range_p_t = 2 * range_Q / 2 * self.dx.mass * 9.806
        n_tau = 14


        if (self.u_prev is None):
            self.u_prev = th.zeros(4, n_batch).to(device=self.device)
            self.u_prev[0, :] = self.dx.mass * 9.806


        # Containers for full solution
        nom_x = th.zeros((n_batch, self.T, self.dx.n_state)).to(self.device)
        nom_u = th.zeros((n_batch, self.T, self.dx.n_ctrl)).to(self.device)
        idx_start = 0

        for idx, sigmoid_cost in enumerate(chunks):
            n_chunk = sigmoid_cost.shape[0]
            idx_end = idx_start + n_chunk
            x_Q = sigmoid_cost[:, :14*self.T].to(self.device)  # these are between 0 and 1 right now
            x_p = sigmoid_cost[:, 14*self.T:].to(self.device)  # these are between 0 and 1 right now

            q_p = x_Q[:, :3*self.T] * range_Q + epsilon
            q_q = x_Q[:, 3*self.T:7*self.T] * range_Q + epsilon
            q_v = x_Q[:, 7*self.T:10*self.T] * range_Q + epsilon
            q_w = x_Q[:, 10*self.T:13*self.T] * range_Q + epsilon
            q_t = x_Q[:, 13*self.T:14*self.T] * range_Q + epsilon

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


            # Run MPC
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


        self.u_prev = nom_u[:,0,:].transpose(0,1)

        self.predictions = th.cat((nom_x, nom_u), dim=2).detach()


        # Return actions from MPC. These actions will be taken into account to create a gaussian distribution.
        # Units of first control input are thrust normalized by mass
        thrust = nom_u[:, 0, 0]/self.dx.mass
        # The other 3 control inputs are the body rates, in rad/s
        omegas = nom_u[:,0,1:4]

        # Now we normalize the units, since the simulation environment later will unnormalize them by default
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
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        self.log_std.data.fill_(-100.0)
        self.log_std.requires_grad = False

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)