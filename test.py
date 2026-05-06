import os
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from acmpc_public.training_modules.mlp_only_policy import MlpOnlyPolicy
from acmpc_public.training_modules.mlp_mpc_policy import MlpMpcPolicy

from datetime import datetime


class EpisodeStatsCallback(BaseCallback):
    """仅打印训练关键统计：episode、reward、终止条件、步数、位置和最优回报。"""

    def __init__(self):
        super().__init__()
        self.episode_count = 0
        self.best_reward = -np.inf
        self.best_episode = 0
        self._episode_returns = None
        self._episode_steps = None

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float32)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])

        if rewards.size == 0 or dones.size == 0:
            return True

        if self._episode_returns is None:
            self._episode_returns = np.zeros_like(rewards, dtype=np.float32)
            self._episode_steps = np.zeros_like(rewards, dtype=np.int32)

        self._episode_returns += rewards
        self._episode_steps += 1

        for idx, done in enumerate(dones):
            if not done:
                continue

            self.episode_count += 1
            last_reward = float(self._episode_returns[idx])
            last_steps = int(self._episode_steps[idx])
            info = infos[idx] if idx < len(infos) else {}

            done_reason = info.get("done_reason", "unknown")
            position = info.get("position", [np.nan, np.nan, np.nan])
            pos = np.asarray(position, dtype=np.float32).reshape(-1)
            if pos.size < 3:
                pos = np.pad(pos, (0, 3 - pos.size), mode="constant", constant_values=np.nan)
            pos_str = f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"

            if last_reward > self.best_reward:
                self.best_reward = last_reward
                self.best_episode = self.episode_count

            sep = "=" * 72
            print(f"\n{sep}")
            print(f"episode      : {self.episode_count}")
            print(f"last_reward  : {last_reward:.3f}")
            print(f"done         : {done_reason}")
            print(f"last_steps   : {last_steps}")
            print(f"position     : {pos_str}")
            print(f"best_reward  : {self.best_reward:.3f}")
            print(f"best_episode : {self.best_episode}")
            print(sep)

            self._episode_returns[idx] = 0.0
            self._episode_steps[idx] = 0

        return True


class SimpleLineTrackEnv(gym.Env):
    """
    用于快速验证 PPO 流程的最小连续控制环境。

    目标：
    - 跟踪沿 x 轴负方向匀速运动的目标点。
    - 保持 state 有界且平滑，同时尽量减少控制代价。

    state 布局（10 维）：
    - [0:3]   position（x, y, z）
    - [3:7]   quaternion（w, x, y, z）
    - [7:10]  linear velocity（vx, vy, vz）

    action 布局（4 维，归一化到 [-1, 1]）：
    - [0]     thrust 相关标量
    - [1:4]   机体角速度控制（omega x/y/z）
    """
    def __init__(self, max_steps: int = 200, dt: float = 0.02, target_speed: float = 0.5):
        super().__init__()
        # Episode 最大步数及仿真时间步长。
        self.max_steps = max_steps
        self.dt = dt
        # 目标沿 x 轴负方向的运动速度。
        self.target_speed = target_speed

        # observation_space 包含 position、quaternion 和 velocity。
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )
        # action_space：4 维连续动作，每个维度的物理含义由后续策略代码解释。
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.state = None
        self.step_count = 0

    def _target_position(self, step: int) -> np.ndarray:
        # 目标沿 x 轴匀速运动：x = -target_speed * dt * step。
        return np.array([-self.target_speed * self.dt * step, 0.0, 0.0], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        # 统一输出 float32，与 SB3 期望的 observation 类型保持一致。
        return self.state.astype(np.float32)

    def reset(self, *, seed=None, options=None):
        # 符合 Gymnasium 规范的 reset 接口，返回格式为 (obs, info)。
        super().reset(seed=seed)
        self.step_count = 0

        # 初始化为悬停附近状态：
        # position 位于原点，unit quaternion，velocity 为零。
        self.state = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=np.float32)
        return self._get_obs(), {}

    def step(self, action):
        # 将 action 转为 float32，并裁剪到 action_space 边界内。
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 从 state 中拆出 position、quaternion、velocity。
        pos = self.state[0:3]
        quat = self.state[3:7]
        vel = self.state[7:10]

        # 将归一化 action 映射到简化动力学输入。
        thrust_acc = (action[0] - 0.5) * 2.0
        omega = action[1:4]

        # 极简运动学更新，仅用于快速验证策略流程。
        vel = vel + np.array([-0.6 + thrust_acc * 0.1, 0.0, 0.0], dtype=np.float32) * self.dt
        pos = pos + vel * self.dt

        # 近似更新 quaternion，之后重新归一化以保证数值稳定。
        quat = quat + np.array([0.0, omega[0], omega[1], omega[2]], dtype=np.float32) * 0.01
        quat_norm = np.linalg.norm(quat)
        if quat_norm > 1e-6:
            quat = quat / quat_norm
        else:
            # 数值异常时退回 identity quaternion。
            quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        self.state = np.concatenate([pos, quat, vel]).astype(np.float32)
        self.step_count += 1

        # 构造当前步的参考目标（target position 和期望 velocity）。
        target_pos = self._target_position(self.step_count)
        target_vel = np.array([-self.target_speed, 0.0, 0.0], dtype=np.float32)

        # reward 由三项组成：
        # - 基础分 1.0
        # - 减去 pos_error（位置偏差惩罚，权重 2.0）
        # - 减去 vel_error（速度偏差惩罚，权重 0.5）
        # - 减去 control_cost（动作幅度惩罚，权重 0.05）
        pos_error = np.linalg.norm(pos - target_pos)
        vel_error = np.linalg.norm(vel - target_vel)
        control_cost = 0.05 * np.linalg.norm(action)
        reward = 1.0 - 2.0 * pos_error - 0.5 * vel_error - control_cost

        # Gymnasium 将终止拆分为两个独立标志：
        # - terminated：任务失败/成功（这里为飞出边界）
        # - truncated：时间截断（达到 max_steps）
        terminated = np.linalg.norm(pos) > 5.0
        truncated = self.step_count >= self.max_steps
        info = {
            "target_pos": target_pos,
            "pos_error": pos_error,
            "vel_error": vel_error,
            "position": pos.copy(),
        }
        if terminated:
            info["done_reason"] = "terminated(boundary)"
        elif truncated:
            info["done_reason"] = "truncated(max_steps)"
        return self._get_obs(), float(reward), terminated, truncated, info


POLICY_MODE = os.getenv("POLICY_MODE", "mpc")
TOTAL_TIMESTEPS = int(os.getenv("TOTAL_TIMESTEPS", "20000"))

# 通过环境变量 POLICY_MODE 选择策略实现类。
POLICY_REGISTRY = {
    "mpc": MlpMpcPolicy,
    "mlp": MlpOnlyPolicy,
}


def print_torch_runtime_info() -> None:
    """打印当前 PyTorch/CUDA 运行环境，便于快速确认是否在用 GPU。"""
    # ANSI 颜色在常见终端中可读性更好：INFO 青色、WARNING 黄色、ERROR 红色。
    c_reset = "\033[0m"
    c_info = "\033[96m"
    c_warn = "\033[93m"
    c_error = "\033[91m"
    c_path = "\033[95m"

    torch_version = torch.__version__
    cuda_build = torch.version.cuda
    cuda_available = torch.cuda.is_available()
    print(f"{c_info}[INFO]{c_reset} torch={torch_version}, cuda_build={cuda_build}")

    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"{c_info}[INFO]{c_reset} runtime_device={c_path}cuda:0{c_reset}, gpu={device_name}")
    else:
        print(f"{c_warn}[WARNING]{c_reset} CUDA not available, training will run on CPU")
        print(f"{c_error}[ERROR]{c_reset} Check current env torch build if this is unexpected")


def main() -> None:

    print_torch_runtime_info()
    # 提前校验 POLICY_MODE，避免在训练中途才发现配置错误。
    if POLICY_MODE not in POLICY_REGISTRY:
        raise ValueError(f"Unsupported POLICY_MODE: {POLICY_MODE}")

    # MPC 策略需要通过环境变量 ACMPC_T 指定预测时域长度。
    if POLICY_MODE == "mpc":
        os.environ.setdefault("ACMPC_T", "5")


    # 实例化环境和对应的 policy class。
    env = SimpleLineTrackEnv()
    policy_class = POLICY_REGISTRY[POLICY_MODE]

    # 创建 PPO 模型，参数为轻量级训练验证配置。
    model = PPO(
        policy=policy_class,
        env=env,
        verbose=0,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
    )

    # 按配置的 total_timesteps 执行训练。
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=EpisodeStatsCallback())

    # 按策略名称保存检查点到 outputs 目录。
    output_dir = Path(__file__).resolve().parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    current_time = datetime.now().strftime("%y-%m-%d_%H-%M")

    model.save(output_dir / f"ppo_{POLICY_MODE}_line_track_{current_time}")


if __name__ == "__main__":
    main()