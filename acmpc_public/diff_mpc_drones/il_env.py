import torch
torch.set_printoptions(linewidth=200)

from mpc import mpc
from mpc.mpc import GradMethods, QuadCost
from mpc.dynamics import NNDynamics
import mpc.util as eutil

import numpy as np
import numpy.random as npr

import os
import sys
import shutil
import time

import pickle as pkl

from setproctitle import setproctitle

import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torch import optim
from torch.nn.utils import parameters_to_vector
from scipy import interpolate

import drone

# ==========================================
# 该文件作用：MPC 求解封装层（策略网络与动力学模型之间的桥）
#
# 典型调用链（见 training_modules/mlp_mpc_policy.py）：
# features -> 预测 Q/p -> IL_Env.mpc(...) -> 求解得到 x_mpc/u_mpc
# ==========================================



class IL_Env:
    def __init__(self, env, lqr_iter=500, mpc_T=5, slew_rate_penalty=None):
        # env: 当前只支持 'drone'
        # lqr_iter/mpc_T: iLQR 最大迭代数与预测时域长度
        self.env = env


        if self.env == 'drone':
            self.true_dx = drone.DroneDx()
        else:
            assert False

        self.lqr_iter = lqr_iter
        self.mpc_T = mpc_T
        self.slew_rate_penalty = slew_rate_penalty

        # 梯度模式：使用解析 Jacobian（由 DroneDx.grad_input 提供）
        self.grad_method = GradMethods.ANALYTIC
        # self.grad_method = GradMethods.AUTO_DIFF
        # self.grad_method = GradMethods.FINITE_DIFF
        # self.grad_method = GradMethods.ANALYTIC_CHECK

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.original_trajs = []

        # 缓存 mpc.MPC solver，避免每次调用都重新构造 nn.Module 实例。
        # 键：(n_batch, device_str)，值：mpc.MPC 实例。
        # 若 n_batch 或设备变化才新建，否则复用同一实例并更新 u_init。
        self._solver_cache: dict = {}


    def mpc(self, dx, xinit, Q, p, u_init=None, eps_override=False,
            lqr_iter_override=None):
        # 参数说明：
        # dx      : 动力学对象（通常为 drone.DroneDx）
        # xinit   : 初始状态 [batch, n_state]
        # Q, p    : 时域二次代价 [T,batch,n_tau,n_tau] / [T,batch,n_tau]
        # u_init  : 预期 warm-start 控制序列（见下方实现注意事项）
        n_batch = xinit.shape[0]

        if xinit.is_cuda:
            this_device = "cuda:0"
        else:
            this_device = "cpu"

        # state+control 维度和（当前实现未使用）
        n_sc = self.true_dx.n_state + self.true_dx.n_ctrl

        # p = p.unsqueeze(0).repeat(self.mpc_T, n_batch, 1)

        # 求解容差：优先 override，否则用 DroneDx 默认 mpc_eps
        if eps_override:
            eps = eps_override
        else:
            eps = self.true_dx.mpc_eps

        # 迭代次数：优先 override，否则用构造函数中的默认值
        if lqr_iter_override:
            lqr_iter = lqr_iter_override
        else:
            lqr_iter = self.lqr_iter


        # ===== [新逻辑：启用] =====
        # 优先使用调用方传入的 warm-start；仅在未提供时使用悬停推力初始化。
        if u_init is None:
            u_init_prepared = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
            u_init_prepared[:, :, 0] = self.true_dx.mass * 9.8066
        else:
            u_init_prepared = u_init.to(device=this_device)

            # 兼容不同 warm-start 形状，统一转换到 [T, batch, n_ctrl]
            # 支持：
            # - [T, batch, n_ctrl]
            # - [batch, T, n_ctrl]
            # - [batch, n_ctrl]
            # - [n_ctrl, batch]  (历史实现常见)
            if u_init_prepared.ndim == 3:
                if u_init_prepared.shape == (self.mpc_T, n_batch, self.true_dx.n_ctrl):
                    pass
                elif u_init_prepared.shape == (n_batch, self.mpc_T, self.true_dx.n_ctrl):
                    u_init_prepared = u_init_prepared.transpose(0, 1)
                else:
                    u_init_prepared = None
            elif u_init_prepared.ndim == 2:
                if u_init_prepared.shape == (n_batch, self.true_dx.n_ctrl):
                    u_init_prepared = u_init_prepared.unsqueeze(0).repeat(self.mpc_T, 1, 1)
                elif u_init_prepared.shape == (self.true_dx.n_ctrl, n_batch):
                    u_init_prepared = u_init_prepared.transpose(0, 1).unsqueeze(0).repeat(self.mpc_T, 1, 1)
                else:
                    u_init_prepared = None
            else:
                u_init_prepared = None

            # 非法输入回退到悬停初始化，避免训练中断
            if u_init_prepared is None:
                u_init_prepared = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
                u_init_prepared[:, :, 0] = self.true_dx.mass * 9.8066

        # 取缓存的 MPC solver（或新建后缓存），避免每次调用都构造 nn.Module。
        # 缓存键按 (n_batch, device) 区分，保证形状匹配。
        cache_key = (n_batch, this_device)
        if cache_key not in self._solver_cache:
            # 控制约束（按 [fc, wx, wy, wz]）
            # fc 为总推力，因此上下界是单桨推力界限 * 4
            lower = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
            lower[:, :, 0] = self.true_dx.thrust_min * 4
            lower[:, :, 1] = -self.true_dx.omega_max[0]
            lower[:, :, 2] = -self.true_dx.omega_max[1]
            lower[:, :, 3] = -self.true_dx.omega_max[2]
            upper = torch.zeros((self.mpc_T, n_batch, self.true_dx.n_ctrl)).to(device=this_device)
            upper[:, :, 0] = self.true_dx.thrust_max * 4
            upper[:, :, 1] = self.true_dx.omega_max[0]
            upper[:, :, 2] = self.true_dx.omega_max[1]
            upper[:, :, 3] = self.true_dx.omega_max[2]
            self._solver_cache[cache_key] = mpc.MPC(
                self.true_dx.n_state, self.true_dx.n_ctrl, self.mpc_T,
                u_lower=lower, u_upper=upper, u_init=u_init_prepared,
                lqr_iter=lqr_iter,
                verbose=-1,
                exit_unconverged=False,
                detach_unconverged=False,
                linesearch_decay=self.true_dx.linesearch_decay,
                max_linesearch_iter=self.true_dx.max_linesearch_iter,
                grad_method=self.grad_method,
                eps=eps,
            )
        solver = self._solver_cache[cache_key]
        # 每次调用前更新 warm-start，然后复用同一 solver 实例。
        solver.u_init = u_init_prepared

        # 调用 MPC 求解器：目标 QuadCost(Q,p)，动力学 dx
        x_mpc, u_mpc, objs_mpc = solver(xinit, QuadCost(Q, p), dx)
        # 返回名义状态/控制轨迹（通常是 [T,batch,*]）
        return x_mpc, u_mpc

