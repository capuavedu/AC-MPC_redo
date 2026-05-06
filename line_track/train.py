"""
Real drone dynamics training script for line tracking task.

Task: Navigate from (0,0,0) to target (-5,0,0) with constant velocity 1 m/s.
Duration: 5 seconds, dt=0.02s => 250 steps max.

State: x = [p(3), q(4), v(3)] (10-dim)
Control: u = [f_c, wx, wy, wz] (4-dim, real physical quantities from MPC)
  - f_c: total thrust in Newtons
  - w: body-frame angular velocity in rad/s

Policy modes:
  - 'mpc': Uses MPC with learned cost parameters
  - 'mlp': Uses pure MLP policy
"""

import gc
import os
import sys
import subprocess
import multiprocessing as mp
import warnings
import time
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import acmpc_public
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan

# Suppress known third-party warning from mpc package (torch.uint8 mask deprecation).
warnings.filterwarnings(
    "ignore",
    message=r"indexing with dtype torch\.uint8 is now deprecated, please use a dtype torch\.bool instead\.",
    category=UserWarning,
    module=r"mpc\.util",
)

# Import your policy classes
from acmpc_public.training_modules.mlp_only_policy import MlpOnlyPolicy
from acmpc_public.training_modules.mlp_mpc_policy import MlpMpcPolicy
from acmpc_public.diff_mpc_drones import drone


# Editable in-code defaults. Environment variables with the same names still override these values.
DEFAULT_CONFIG = {
    "POLICY_MODE": "mpc",
    "TOTAL_TIMESTEPS": 2000000,
    "N_ENVS": 8,  # auto-computed when None
    "N_STEPS": 4096,
    "BATCH_SIZE": 256,
    "N_EPOCHS": 8,
    "LEARNING_RATE": 0.0003,
    "VEC_MODE": "subproc",  # subproc | dummy
    "ENV_DEVICE": "cpu",
    "MAX_EPISODE_STEPS": 250,  # 5s horizon at dt=0.02 for the line-track task
}




def _get_cfg(name: str, default_value, cast_fn):
    """Read value from env var first, then fallback to in-code default."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default_value
    return cast_fn(raw)


class EpisodeStatsCallback(BaseCallback):
    """Track and print key episode statistics."""

    def __init__(self, save_best_path: Optional[Path] = None):
        super().__init__()
        self.episode_count = 0
        self.best_reward = -np.inf
        self.best_episode = 0
        self.best_done_reason = "unknown"
        self.best_final_position = np.full(3, np.nan, dtype=np.float32)
        self.save_best_path = save_best_path
        self.best_model_saved = False
        self._episode_returns = None
        self._episode_steps = None

    @staticmethod
    def _read_gpu_utilization() -> str:
        """Read GPU utilization percentage for runtime visibility."""
        if not torch.cuda.is_available():
            return "N/A(CPU)"

        # Prefer direct torch API when available.
        if hasattr(torch.cuda, "utilization"):
            try:
                util = float(torch.cuda.utilization(0))
                return f"{util:.1f}%"
            except Exception:
                pass

        # Fallback to nvidia-smi.
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
                timeout=1.0,
            )
            first_line = output.decode("utf-8", errors="ignore").strip().splitlines()[0]
            util = float(first_line)
            return f"{util:.1f}%"
        except Exception:
            return "N/A(unavailable)"

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", []), dtype=np.float32)
        dones = np.asarray(self.locals.get("dones", []), dtype=bool)
        infos = self.locals.get("infos", [])

        if rewards.size == 0 or dones.size == 0:
            return True
        rewards = np.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        if self._episode_returns is None:
            self._episode_returns = np.zeros_like(rewards, dtype=np.float32)
            self._episode_steps = np.zeros_like(rewards, dtype=np.int32)

        assert self._episode_returns is not None
        assert self._episode_steps is not None

        self._episode_returns += rewards
        self._episode_steps += 1

        ended_rewards = []
        ended_reasons = []
        ended_steps = []
        ended_positions = []

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

            ended_rewards.append(last_reward)
            ended_reasons.append(done_reason)
            ended_steps.append(last_steps)
            ended_positions.append(pos)

            if last_reward > self.best_reward:
                self.best_reward = last_reward
                self.best_episode = self.episode_count
                self.best_done_reason = done_reason
                self.best_final_position = pos.copy()
                if self.save_best_path is not None:
                    # Save a checkpoint each time a new best episode return is observed.
                    self.model.save(str(self.save_best_path))
                    self.best_model_saved = True

            self._episode_returns[idx] = 0.0
            self._episode_steps[idx] = 0

        if ended_rewards:
            ended_rewards_np = np.asarray(ended_rewards, dtype=np.float32)
            ended_steps_np = np.asarray(ended_steps, dtype=np.int32)

            round_avg_reward = float(np.mean(ended_rewards_np))
            round_best_idx = int(np.argmax(ended_rewards_np))
            round_best_reward = float(ended_rewards_np[round_best_idx])
            round_reason = ended_reasons[round_best_idx]
            round_steps = int(ended_steps_np[round_best_idx])
            round_pos = ended_positions[round_best_idx]
            gpu_util = self._read_gpu_utilization()

            sep = "=" * 80
            print(f"\n{sep}")
            print(f"episode                     : {self.episode_count}")
            print(f"last_round_avg_reward       : {round_avg_reward:.4f}")
            print(f"last_round_best_reward      : {round_best_reward:.4f}")
            print(f"last_round_done_reason      : {round_reason}")
            print(f"last_round_done_steps       : {round_steps}")
            print(f"last_round_final_pos        : [{round_pos[0]:.3f}, {round_pos[1]:.3f}, {round_pos[2]:.3f}]")
            print(f"global_best_reward          : {self.best_reward:.4f} (episode {self.best_episode})")

            print(f"gpu_utilization             : {gpu_util}")
            print(sep)

        return True


def _choose_batch_size(rollout_size: int, preferred_batch: int) -> int:
    """Choose a PPO minibatch size that divides rollout_size."""
    candidate = max(1, min(preferred_batch, rollout_size))
    while rollout_size % candidate != 0 and candidate > 1:
        candidate -= 1
    return max(1, candidate)


def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Return a unit quaternion, falling back to identity if needed."""
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat / norm


def plan_straight_t_profile_trajectory(
    start_state: np.ndarray,
    goal_state: np.ndarray,
    max_steps: int,
    dt: float,
    accel_fraction: float = 0.25,
    decel_fraction: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Plan a straight-line rest-to-rest trajectory with a T-shaped speed profile.

    The planner uses the position and attitude components from the provided states.
    Velocity references are generated internally so the trajectory starts and ends at rest.
    """
    start_state = np.asarray(start_state, dtype=np.float32).reshape(10)
    goal_state = np.asarray(goal_state, dtype=np.float32).reshape(10)

    total_time = max(float(max_steps) * float(dt), float(dt))
    accel_fraction = float(np.clip(accel_fraction, 1e-3, 0.499))
    decel_fraction = float(np.clip(decel_fraction, 1e-3, 0.499))
    cruise_fraction = 1.0 - accel_fraction - decel_fraction
    if cruise_fraction < 0.0:
        accel_fraction = 0.5
        decel_fraction = 0.5
        cruise_fraction = 0.0

    accel_time = total_time * accel_fraction
    decel_time = total_time * decel_fraction
    cruise_time = total_time * cruise_fraction

    times = np.arange(max_steps + 1, dtype=np.float32) * float(dt)
    start_pos = start_state[0:3]
    goal_pos = goal_state[0:3]
    delta_pos = goal_pos - start_pos
    distance = float(np.linalg.norm(delta_pos))

    start_quat = _normalize_quaternion(start_state[3:7])
    goal_quat = _normalize_quaternion(goal_state[3:7])

    positions = np.zeros((max_steps + 1, 3), dtype=np.float32)
    quaternions = np.zeros((max_steps + 1, 4), dtype=np.float32)
    velocities = np.zeros((max_steps + 1, 3), dtype=np.float32)

    if distance < 1e-6:
        positions[:] = start_pos
        velocities[:] = 0.0
    else:
        direction = delta_pos / distance
        denom = max(accel_time * (accel_time + cruise_time), 1e-6)
        accel = distance / denom
        vmax = accel * accel_time

        for idx, t in enumerate(times):
            clamped_t = min(float(t), total_time)
            if clamped_t <= accel_time:
                scalar_pos = 0.5 * accel * clamped_t * clamped_t
                scalar_vel = accel * clamped_t
            elif clamped_t <= accel_time + cruise_time:
                scalar_pos = 0.5 * accel * accel_time * accel_time + vmax * (clamped_t - accel_time)
                scalar_vel = vmax
            else:
                remaining = max(total_time - clamped_t, 0.0)
                scalar_pos = distance - 0.5 * accel * remaining * remaining
                scalar_vel = accel * remaining

            positions[idx] = start_pos + direction * scalar_pos
            velocities[idx] = direction * scalar_vel

    blend = np.linspace(0.0, 1.0, max_steps + 1, dtype=np.float32)
    for idx, alpha in enumerate(blend):
        quaternions[idx] = _normalize_quaternion((1.0 - alpha) * start_quat + alpha * goal_quat)

    positions[-1] = goal_pos
    velocities[-1] = np.zeros(3, dtype=np.float32)
    quaternions[-1] = goal_quat

    states = np.concatenate([positions, quaternions, velocities], axis=1).astype(np.float32)
    return {
        "time": times,
        "position": positions,
        "velocity": velocities,
        "quaternion": quaternions,
        "state": states,
    }


class RealDroneLineTrackEnv(gym.Env):
    """
    Real drone dynamics environment for line tracking task.

    Task: Fly from a start hover state to a goal hover state using a straight-line
    T-profile reference trajectory.
    
    State & Control:
    - state x: [p(3), q(4), v(3)]  -- position, quaternion, velocity (10-dim)
    - action u: [f_c, wx, wy, wz] -- thrust, body angular velocities (4-dim)
    
    Internally stores state as torch.Tensor for efficient dynamics computation.
    Gym interface (reset/step) uses numpy arrays.
    """

    def __init__(
        self,
        max_steps: int = 250,  # 5s at dt=0.02s
        dt: float = 0.02,
        target_distance: float = 5.0,  # meters
        start_state: Optional[np.ndarray] = None,
        goal_state: Optional[np.ndarray] = None,
        device: str = 'cpu'
    ):
        super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.target_distance = target_distance
        self.device = torch.device(device)

        default_start_state = np.array([
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=np.float32)
        default_goal_state = np.array([
            -target_distance, 0.0, 0.0,
            1.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ], dtype=np.float32)
        self.start_state_np = np.asarray(
            default_start_state if start_state is None else start_state,
            dtype=np.float32,
        ).reshape(10)
        self.goal_state_np = np.asarray(
            default_goal_state if goal_state is None else goal_state,
            dtype=np.float32,
        ).reshape(10)

        # Real drone dynamics (torch-native)
        self.dx = drone.DroneDx(device=str(device))

        # Observation: position, quaternion, velocity (10-dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        # Action re-parameterization:
        # - a[0] in [-1, 1] is normalized thrust delta around hover thrust.
        # - a[1:4] are body angular velocities in physical units.
        # This makes initial policy output (near zero) correspond to near-hover control.
        self.hover_thrust = float(-self.dx.g[2].item() * self.dx.mass)
        self.thrust_phys_low = float(self.dx.thrust_min * 4)
        self.thrust_phys_high = float(self.dx.thrust_max * 4)
        self.thrust_delta_limit = float(min(
            self.hover_thrust - self.thrust_phys_low,
            self.thrust_phys_high - self.hover_thrust,
        ))

        # Action: [normalized thrust delta, wx, wy, wz]
        self.action_space = spaces.Box(
            low=np.array([
                -1.0,
                float(-self.dx.omega_max[0]),
                float(-self.dx.omega_max[1]),
                float(-self.dx.omega_max[2]),
            ], dtype=np.float32),
            high=np.array([
                1.0,
                float(self.dx.omega_max[0]),
                float(self.dx.omega_max[1]),
                float(self.dx.omega_max[2]),
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # Policy-side bounds (for clipping incoming actions)
        self.u_low = torch.tensor(
            [-1.0,
             float(-self.dx.omega_max[0]),
             float(-self.dx.omega_max[1]),
             float(-self.dx.omega_max[2])],
            dtype=torch.float32, device=self.device)
        self.u_high = torch.tensor(
            [1.0,
             float(self.dx.omega_max[0]),
             float(self.dx.omega_max[1]),
             float(self.dx.omega_max[2])],
            dtype=torch.float32, device=self.device)

        # Physical control bounds for dynamics input u=[fc, wx, wy, wz]
        self.u_phys_low = torch.tensor(
            [self.thrust_phys_low,
             float(-self.dx.omega_max[0]),
             float(-self.dx.omega_max[1]),
             float(-self.dx.omega_max[2])],
            dtype=torch.float32, device=self.device)
        self.u_phys_high = torch.tensor(
            [self.thrust_phys_high,
             float(self.dx.omega_max[0]),
             float(self.dx.omega_max[1]),
             float(self.dx.omega_max[2])],
            dtype=torch.float32, device=self.device)

        trajectory = plan_straight_t_profile_trajectory(
            start_state=self.start_state_np,
            goal_state=self.goal_state_np,
            max_steps=self.max_steps,
            dt=self.dt,
        )
        self.reference_state = torch.as_tensor(trajectory["state"], dtype=torch.float32, device=self.device)
        self.reference_pos = torch.as_tensor(trajectory["position"], dtype=torch.float32, device=self.device)
        self.reference_vel = torch.as_tensor(trajectory["velocity"], dtype=torch.float32, device=self.device)
        self.reference_quat = torch.as_tensor(trajectory["quaternion"], dtype=torch.float32, device=self.device)

        # Internal state: torch tensor for efficient computation
        self.state = None  # torch.Tensor [10]
        self.step_count = 0

    def _reference_index(self, step: int) -> int:
        return int(np.clip(step, 0, self.max_steps))

    def _reference_at(self, step: int):
        idx = self._reference_index(step)
        return self.reference_state[idx], self.reference_pos[idx], self.reference_vel[idx], self.reference_quat[idx]

    def get_reference(self, step: int) -> Dict[str, np.ndarray]:
        idx = self._reference_index(step)
        return {
            "state": self.reference_state[idx].detach().cpu().numpy().copy(),
            "position": self.reference_pos[idx].detach().cpu().numpy().copy(),
            "velocity": self.reference_vel[idx].detach().cpu().numpy().copy(),
            "quaternion": self.reference_quat[idx].detach().cpu().numpy().copy(),
        }

    def _get_obs(self) -> np.ndarray:
        """Convert internal torch tensor to numpy for gym interface."""
        assert self.state is not None
        obs = self.state.detach().cpu().numpy().astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        if options is not None:
            start_state = options.get("start_state")
            goal_state = options.get("goal_state")
            if start_state is not None or goal_state is not None:
                self.start_state_np = np.asarray(
                    self.start_state_np if start_state is None else start_state,
                    dtype=np.float32,
                ).reshape(10)
                self.goal_state_np = np.asarray(
                    self.goal_state_np if goal_state is None else goal_state,
                    dtype=np.float32,
                ).reshape(10)
                trajectory = plan_straight_t_profile_trajectory(
                    start_state=self.start_state_np,
                    goal_state=self.goal_state_np,
                    max_steps=self.max_steps,
                    dt=self.dt,
                )
                self.reference_state = torch.as_tensor(trajectory["state"], dtype=torch.float32, device=self.device)
                self.reference_pos = torch.as_tensor(trajectory["position"], dtype=torch.float32, device=self.device)
                self.reference_vel = torch.as_tensor(trajectory["velocity"], dtype=torch.float32, device=self.device)
                self.reference_quat = torch.as_tensor(trajectory["quaternion"], dtype=torch.float32, device=self.device)

        self.state = torch.as_tensor(self.start_state_np, dtype=torch.float32, device=self.device).clone()
        self.state = torch.nan_to_num(self.state, nan=0.0, posinf=0.0, neginf=0.0)
        quat = self.state[3:7]
        quat_norm = quat.norm()
        if bool(torch.isfinite(quat_norm)) and float(quat_norm.item()) > 1e-6:
            self.state[3:7] = quat / quat_norm
        else:
            self.state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)

        ref = self.get_reference(step=0)
        info = {
            "target_pos": ref["position"],
            "target_vel": ref["velocity"],
            "target_quat": ref["quaternion"],
        }

        return self._get_obs(), info

    def step(self, action):
        assert self.state is not None

        # Convert incoming action (numpy from SB3) to torch immediately.
        # a[0] is normalized thrust delta, mapped to physical thrust around hover.
        a = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        a = a.clamp(self.u_low, self.u_high)

        thrust_delta_norm = a[0]
        thrust_cmd = self.hover_thrust + thrust_delta_norm * self.thrust_delta_limit
        u = torch.cat([thrust_cmd.unsqueeze(0), a[1:4]])
        u = u.clamp(self.u_phys_low, self.u_phys_high)

        # Advance dynamics: fully torch, no numpy conversion
        with torch.no_grad():
            x_next = self.dx.forward(self.state.unsqueeze(0), u.unsqueeze(0))  # [1, 10]
        x_next = x_next.squeeze(0)
        state_has_nonfinite = not bool(torch.isfinite(x_next).all().item())
        if state_has_nonfinite:
            # Keep the previous state as a safe fallback when dynamics diverges.
            x_next = self.state.clone()
        self.state = torch.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0)
        quat = self.state[3:7]
        quat_norm = quat.norm()
        if bool(torch.isfinite(quat_norm)) and float(quat_norm.item()) > 1e-6:
            self.state[3:7] = quat / quat_norm
        else:
            self.state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        self.step_count += 1

        # Slice state (still on device)
        pos = self.state[0:3]
        quat = self.state[3:7]
        vel = self.state[7:10]

        # Reference trajectory (torch tensors on device)
        _, target_pos, target_vel, target_quat = self._reference_at(self.step_count)

        # Reward: all torch arithmetic, scalar via .item()
        pos_error = (pos - target_pos).norm()
        vel_error = (vel - target_vel).norm()
        q_error = 1.0 - torch.abs(torch.dot(quat, target_quat))
        # Penalize control effort relative to hover, not absolute thrust.
        control_vec = torch.cat([thrust_delta_norm.unsqueeze(0), a[1:4] / self.u_high[1:4]])
        control_cost = 0.01 * control_vec.norm()
        reward = 5.0 - 3.0 * pos_error  - 0.5*vel_error - 0.1*q_error - control_cost
        reward = torch.nan_to_num(reward, nan=-100.0, posinf=100.0, neginf=-100.0).clamp(-200.0, 200.0)

        # Terminal conditions (scalar bool)
        terminated = bool(pos.norm().item() > 15.0) or state_has_nonfinite
        truncated = self.step_count >= self.max_steps

        # Convert to numpy only at the gym boundary (info dict consumed externally)
        pos_np = np.nan_to_num(pos.detach().cpu().numpy(), nan=0.0, posinf=1e3, neginf=-1e3)
        info = {
            "position": pos_np.copy(),
            "target_pos": np.nan_to_num(target_pos.detach().cpu().numpy(), nan=0.0, posinf=1e3, neginf=-1e3).copy(),
            "target_vel": np.nan_to_num(target_vel.detach().cpu().numpy(), nan=0.0, posinf=1e3, neginf=-1e3).copy(),
            "target_quat": np.nan_to_num(target_quat.detach().cpu().numpy(), nan=0.0, posinf=1.0, neginf=-1.0).copy(),
            "pos_error": float(torch.nan_to_num(pos_error, nan=1e3, posinf=1e3, neginf=1e3).item()),
            "vel_error": float(torch.nan_to_num(vel_error, nan=1e3, posinf=1e3, neginf=1e3).item()),
            "quat_error": float(torch.nan_to_num(q_error, nan=1.0, posinf=1.0, neginf=1.0).item()),
            "velocity": np.nan_to_num(vel.detach().cpu().numpy(), nan=0.0, posinf=1e3, neginf=-1e3).copy(),
            "quaternion": np.nan_to_num(quat.detach().cpu().numpy(), nan=0.0, posinf=1.0, neginf=-1.0).copy(),
            "action_raw": np.nan_to_num(a.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).copy(),
            "action_physical": np.nan_to_num(u.detach().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0).copy(),
            "hover_thrust": self.hover_thrust,
            "numerical_issue": state_has_nonfinite,
        }

        if terminated:
            info["done_reason"] = (
                "terminated(numerical_issue)" if state_has_nonfinite else "terminated(out_of_bounds)"
            )
        elif truncated:
            info["done_reason"] = "truncated(max_steps)"
        else:
            info["done_reason"] = "in_progress"

        return self._get_obs(), float(reward.item()), terminated, truncated, info


def print_torch_runtime_info() -> None:
    """Print PyTorch/CUDA runtime information."""
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


class _TeeStream:
    """Mirror stdout/stderr to both console and a log file."""

    def __init__(self, console_stream, log_file_handle):
        self._console_stream = console_stream
        self._log_file_handle = log_file_handle

    def write(self, data):
        self._console_stream.write(data)
        self._log_file_handle.write(data)

    def flush(self):
        self._console_stream.flush()
        self._log_file_handle.flush()


def _setup_run_logging(run_dir: Path):
    """Redirect stdout/stderr to both terminal and run-local log file."""
    log_path = run_dir / "training_log.txt"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)
    return log_file, original_stdout, original_stderr, log_path


def main():
    # Configuration from in-code defaults with optional env-var override.
    policy_mode = _get_cfg("POLICY_MODE", DEFAULT_CONFIG["POLICY_MODE"], str).lower()
    total_timesteps = _get_cfg("TOTAL_TIMESTEPS", DEFAULT_CONFIG["TOTAL_TIMESTEPS"], int)
    model_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_n_envs = max(1, min(8, mp.cpu_count() // 2 if mp.cpu_count() > 1 else 1))
    n_envs_default = default_n_envs if DEFAULT_CONFIG["N_ENVS"] is None else int(DEFAULT_CONFIG["N_ENVS"])
    n_envs = _get_cfg("N_ENVS", n_envs_default, int)
    n_steps = _get_cfg("N_STEPS", DEFAULT_CONFIG["N_STEPS"], int)
    max_episode_steps = _get_cfg("MAX_EPISODE_STEPS", DEFAULT_CONFIG["MAX_EPISODE_STEPS"], int)
    preferred_batch_size = _get_cfg("BATCH_SIZE", DEFAULT_CONFIG["BATCH_SIZE"], int)
    n_epochs = _get_cfg("N_EPOCHS", DEFAULT_CONFIG["N_EPOCHS"], int)
    learning_rate = _get_cfg("LEARNING_RATE", DEFAULT_CONFIG["LEARNING_RATE"], float)
    vec_mode = _get_cfg("VEC_MODE", DEFAULT_CONFIG["VEC_MODE"], str).lower()  # subproc | dummy
    env_device = _get_cfg("ENV_DEVICE", DEFAULT_CONFIG["ENV_DEVICE"], str).lower()

    # Create output directory early so all runtime messages can be captured to log.
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%m%d%H%M")
    model_name = f"ppo_{policy_mode}_{current_time}"
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    original_stdout = None
    original_stderr = None
    log_path = None

    # 定义资源监控函数（延迟导入以避免顶层依赖）
    def log_resource_usage(tag=""):
        try:
            import psutil
            import pynvml
            pynvml.nvmlInit()
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            vms_mb = mem_info.vms / 1024 / 1024
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem_used = mem.used / 1024 / 1024
                gpu_mem_total = mem.total / 1024 / 1024
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                gpu_info = f"GPU: {gpu_mem_used:.1f}/{gpu_mem_total:.1f} MB, Util: {gpu_util}%"
            except Exception:
                gpu_info = "GPU: unavailable"
            print(f"[RESOURCE]{tag} RAM: {rss_mb:.1f}MB RSS, {vms_mb:.1f}MB VMS | {gpu_info}")
        except Exception as e:
            print(f"[RESOURCE] failed: {e}")

    # 计算训练相关配置
    rollout_size = n_steps * n_envs
    batch_size = _choose_batch_size(rollout_size=rollout_size, preferred_batch=preferred_batch_size)
    minibatches_per_epoch = rollout_size // batch_size
    updates_per_episode_est = float(max_episode_steps) / float(n_steps)
    minibatches_per_episode_est = updates_per_episode_est * minibatches_per_epoch * n_epochs

    # 重定向所有输出到日志文件（同时保留终端输出）
    log_file, original_stdout, original_stderr, log_path = _setup_run_logging(run_dir)

    # 确保 MPC 相关环境变量在构建策略前存在
    if policy_mode == "mpc":
        os.environ.setdefault("ACMPC_T", "2")

    log_resource_usage(" [BEFORE TRAIN]")

    # ...原有配置打印...
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Policy Mode       : {policy_mode}")
    print(f"Total Timesteps   : {total_timesteps}")
    print(f"Model Device      : {model_device}")
    print(f"Env Device        : {env_device}")
    print(f"Vec Mode          : {vec_mode}")
    print(f"N Env             : {n_envs}")
    print(f"N Steps           : {n_steps}")
    print(f"Max Episode Steps : {max_episode_steps}")
    print(f"Rollout Size      : {rollout_size}")
    print(f"Batch Size        : {batch_size}")
    print(f"N Epochs          : {n_epochs}")
    print(f"Learning Rate     : {learning_rate}")
    print(f"MiniBatch/Epoch   : {minibatches_per_epoch}")
    print(f"Est Batch/Episode : {minibatches_per_episode_est:.2f}")
    if policy_mode == "mpc":
        print(f"MPC Horizon (T)   : {os.environ.get('ACMPC_T', '2')}")
    print(f"Log Path          : {log_path}")
    print(f"{'='*80}\n")

    # Policy registry
    policy_registry = {
        "mpc": MlpMpcPolicy,
        "mlp": MlpOnlyPolicy,
    }

    if policy_mode not in policy_registry:
        raise ValueError(f"Unsupported POLICY_MODE: {policy_mode}. Choose from {list(policy_registry.keys())}")

    # Create vectorized environment and policy
    vec_env_cls = SubprocVecEnv if vec_mode == "subproc" else DummyVecEnv
    env = make_vec_env(
        RealDroneLineTrackEnv,
        n_envs=n_envs,
        seed=0,
        env_kwargs={"device": env_device, "max_steps": max_episode_steps},
        vec_env_cls=vec_env_cls,
    )
    env = VecCheckNan(env, raise_exception=False, warn_once=True)
    policy_class = policy_registry[policy_mode]

    # Create PPO model.
    # max_grad_norm=0.5 is already SB3's default, but we set it explicitly
    # because Inf gradients (from near-zero std) interact badly with the clipping:
    #   clip_coef = 0.5 / Inf = 0  →  grad * 0 = NaN.
    # The real guard is log_std=-2 in MlpMpcPolicy; this is defense-in-depth.
    model = PPO(
        policy=policy_class,
        env=env,
        verbose=0,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        gamma=0.99,
        clip_range=0.2,
        max_grad_norm=0.5,
        device=model_device,
    )

    # Train
    print(f"Starting training for {total_timesteps} timesteps...\n")
    train_start_time = time.perf_counter()
    episode_stats_callback = EpisodeStatsCallback(save_best_path=run_dir / "policy_best")

    # 训练时每N步记录一次资源占用
    class ResourceLogCallback(BaseCallback):
        def __init__(self, log_interval=5000):
            super().__init__()
            self.log_interval = log_interval
            self.last_log = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last_log >= self.log_interval:
                log_resource_usage(f" [TRAIN step={self.num_timesteps}]")
                self.last_log = self.num_timesteps
            return True

    # 每个 rollout 结束后强制 GC，回收 mpc.MPC autograd Function 产生的循环引用。
    # 这些对象在 Python 引用计数下无法自动释放（因为 computation graph 存在环），
    # 不显式 gc.collect() + empty_cache() 会导致 RAM/GPU 每 rollout 增长 ~600MB。
    class GCCallback(BaseCallback):
        """Force cycle GC + CUDA cache flush after every rollout to prevent memory leak."""

        def __init__(self, rollout_size: int):
            super().__init__()
            self._rollout_size = max(1, rollout_size)
            self._last_gc_step = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self._last_gc_step >= self._rollout_size:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._last_gc_step = self.num_timesteps
            return True

    resource_callback = ResourceLogCallback(log_interval=1000)
    gc_callback = GCCallback(rollout_size=rollout_size)
    from stable_baselines3.common.callbacks import CallbackList
    callbacks = CallbackList([episode_stats_callback, resource_callback, gc_callback])

    model.learn(total_timesteps=total_timesteps, callback=callbacks)
    train_elapsed_seconds = time.perf_counter() - train_start_time

    log_resource_usage(" [AFTER TRAIN]")

    best_pos = episode_stats_callback.best_final_position
    print(f"\n{'='*80}")
    print("Final Best Episode Summary")
    print(f"{'='*80}")
    print(f"best_reward                : {episode_stats_callback.best_reward:.4f}")
    print(f"best_episode               : {episode_stats_callback.best_episode}")
    print(f"best_done_reason           : {episode_stats_callback.best_done_reason}")
    print(
        "best_final_position        : "
        f"[{best_pos[0]:.3f}, {best_pos[1]:.3f}, {best_pos[2]:.3f}]"
    )
    print(f"train_elapsed_seconds      : {train_elapsed_seconds:.2f}")
    print(f"train_elapsed_hms          : {time.strftime('%H:%M:%S', time.gmtime(train_elapsed_seconds))}")
    print(f"{'='*80}\n")

    # Save final model snapshot (the last parameters after all updates).
    model.save(run_dir / "policy")
    if not episode_stats_callback.best_model_saved:
        # Fallback for very short runs where no episode completed.
        model.save(run_dir / "policy_best")

    metadata = {
        "policy_mode": policy_mode,
        "total_timesteps": total_timesteps,
        "timestamp": current_time,
        "timestamp_format": "MMDDHHMM",
        "max_episode_steps": max_episode_steps,
        "target_distance": 5.0,
        "trajectory_type": "straight_t_profile",
        "start_state": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "goal_state": [-5.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "model_device": model_device,
        "env_device": env_device,
        "n_envs": n_envs,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "learning_rate": learning_rate,
        "vec_mode": vec_mode,
        "saved_models": {
            "final": "policy.zip",
            "best": "policy_best.zip",
        },
        "best_training_episode": int(episode_stats_callback.best_episode),
        "best_training_reward": float(episode_stats_callback.best_reward),
        "training_time_seconds": float(train_elapsed_seconds),
        "training_time_hms": time.strftime('%H:%M:%S', time.gmtime(train_elapsed_seconds)),
        "training_log_file": "training_log.txt",
    }

    import json
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training Complete")
    print(f"{'='*80}")
    print(f"Model saved to: {run_dir}")
    print(f"Training log  : {log_path}")
    print(f"{'='*80}\n")
    # 清理（恢复 stdout/stderr 并关闭日志文件）
    if log_file is not None:
        try:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        finally:
            log_file.close()


if __name__ == "__main__":
    # 检查依赖
    try:
        import psutil
    except ImportError:
        print("[ERROR] 缺少psutil库，请先运行: pip install psutil 或 conda install psutil")
        sys.exit(1)
    try:
        import pynvml
    except ImportError:
        print("[ERROR] 缺少nvidia-ml-py3库，请先运行: pip install nvidia-ml-py3 或 conda install -c conda-forge nvidia-ml-py3")
        sys.exit(1)

    main()
