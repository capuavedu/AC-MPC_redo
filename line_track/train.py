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

import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path so we can import acmpc_public
sys.path.insert(0, str(Path(__file__).parent.parent))

import gym
import numpy as np
from gym import spaces
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# Import your policy classes
from acmpc_public.training_modules.mlp_only_policy import MlpOnlyPolicy
from acmpc_public.training_modules.mlp_mpc_policy import MlpMpcPolicy
from acmpc_public.diff_mpc_drones import drone


class EpisodeStatsCallback(BaseCallback):
    """Track and print key episode statistics."""

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

            sep = "=" * 80
            print(f"\n{sep}")
            print(f"episode      : {self.episode_count}")
            print(f"last_reward  : {last_reward:.4f}")
            print(f"done_reason  : {done_reason}")
            print(f"last_steps   : {last_steps}")
            print(f"final_pos    : {pos_str}")
            print(f"best_reward  : {self.best_reward:.4f} (episode {self.best_episode})")
            print(sep)

            self._episode_returns[idx] = 0.0
            self._episode_steps[idx] = 0

        return True


class RealDroneLineTrackEnv(gym.Env):
    """
    Real drone dynamics environment for line tracking task.
    
    Task: Track a reference trajectory moving at constant velocity along -X axis.
    - Start position: (0, 0, 0)
    - Target position: (-5, 0, 0) after 5 seconds
    - Reference velocity: 1.0 m/s along -X axis
    
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
        target_speed: float = 1.0,  # m/s
        device: str = 'cpu'
    ):
        super().__init__()
        self.max_steps = max_steps
        self.dt = dt
        self.target_distance = target_distance
        self.target_speed = target_speed
        self.device = torch.device(device)

        # Real drone dynamics (torch-native)
        self.dx = drone.DroneDx(device=str(device))

        # Observation: position, quaternion, velocity (10-dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        # Action: thrust, body angular velocities (4-dim, real physical units)
        self.action_space = spaces.Box(
            low=np.array([
                float(self.dx.thrust_min * 4),
                float(-self.dx.omega_max[0]),
                float(-self.dx.omega_max[1]),
                float(-self.dx.omega_max[2]),
            ], dtype=np.float32),
            high=np.array([
                float(self.dx.thrust_max * 4),
                float(self.dx.omega_max[0]),
                float(self.dx.omega_max[1]),
                float(self.dx.omega_max[2]),
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # Cache action bounds as torch tensors — avoids re-creating them every step
        self.u_low = torch.tensor(
            [float(self.dx.thrust_min * 4),
             float(-self.dx.omega_max[0]),
             float(-self.dx.omega_max[1]),
             float(-self.dx.omega_max[2])],
            dtype=torch.float32, device=self.device)
        self.u_high = torch.tensor(
            [float(self.dx.thrust_max * 4),
             float(self.dx.omega_max[0]),
             float(self.dx.omega_max[1]),
             float(self.dx.omega_max[2])],
            dtype=torch.float32, device=self.device)

        # Reference velocity tensor (constant for line-track task)
        self.target_vel = torch.tensor(
            [-self.target_speed, 0.0, 0.0],
            dtype=torch.float32, device=self.device)

        # Internal state: torch tensor for efficient computation
        self.state = None  # torch.Tensor [10]
        self.step_count = 0

    def _target_position(self, step: int) -> torch.Tensor:
        """Reference position at timestep k: linearly moving along -X."""
        x = -self.target_distance * min(step * self.dt / (self.max_steps * self.dt), 1.0)
        return torch.tensor([x, 0.0, 0.0], dtype=torch.float32, device=self.device)

    def _get_obs(self) -> np.ndarray:
        """Convert internal torch tensor to numpy for gym interface."""
        return self.state.cpu().numpy().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        # Initial state: at origin, identity quaternion, stationary
        # Store as torch tensor on device for efficient computation
        self.state = torch.tensor([
            0.0, 0.0, 0.0,           # position
            1.0, 0.0, 0.0, 0.0,      # quaternion (identity)
            0.0, 0.0, 0.0,           # velocity
        ], dtype=torch.float32, device=self.device)

        return self._get_obs(), {}

    def step(self, action):
        # Convert incoming action (numpy from SB3) to torch immediately
        u = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        # Clip to physical bounds (torch, no numpy involved)
        u = u.clamp(self.u_low, self.u_high)

        # Advance dynamics: fully torch, no numpy conversion
        with torch.no_grad():
            x_next = self.dx.forward(self.state.unsqueeze(0), u.unsqueeze(0))  # [1, 10]
        self.state = x_next.squeeze(0)
        self.step_count += 1

        # Slice state (still on device)
        pos = self.state[0:3]
        quat = self.state[3:7]
        vel = self.state[7:10]

        # Reference trajectory (torch tensors on device)
        target_pos = self._target_position(self.step_count)

        # Reward: all torch arithmetic, scalar via .item()
        pos_error = (pos - target_pos).norm()
        vel_error = (vel - self.target_vel).norm()
        q_error = 1.0 - quat[0].abs()          # prefer identity quaternion
        control_cost = 0.01 * u.norm()
        reward = 1.0 - 3.0 * pos_error - 0.5 * vel_error - 0.3 * q_error - control_cost

        # Terminal conditions (scalar bool)
        terminated = bool(pos.norm().item() > 10.0)
        truncated = self.step_count >= self.max_steps

        # Convert to numpy only at the gym boundary (info dict consumed externally)
        pos_np = pos.cpu().numpy()
        info = {
            "position": pos_np.copy(),
            "target_pos": target_pos.cpu().numpy().copy(),
            "pos_error": pos_error.item(),
            "vel_error": vel_error.item(),
            "velocity": vel.cpu().numpy().copy(),
            "quaternion": quat.cpu().numpy().copy(),
            "action": u.cpu().numpy().copy(),
        }

        if terminated:
            info["done_reason"] = "terminated(out_of_bounds)"
        elif truncated:
            info["done_reason"] = "truncated(max_steps)"
        else:
            info["done_reason"] = "in_progress"

        return self._get_obs(), reward.item(), terminated, truncated, info


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


def main():
    print_torch_runtime_info()

    # Configuration from environment variables or defaults
    policy_mode = os.getenv("POLICY_MODE", "mpc").lower()
    total_timesteps = int(os.getenv("TOTAL_TIMESTEPS", "100000"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # For MPC policy, set prediction horizon
    if policy_mode == "mpc":
        os.environ.setdefault("ACMPC_T", "5")

    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Policy Mode       : {policy_mode}")
    print(f"Total Timesteps   : {total_timesteps}")
    print(f"Device            : {device}")
    if policy_mode == "mpc":
        print(f"MPC Horizon (T)   : {os.environ.get('ACMPC_T', '5')}")
    print(f"{'='*80}\n")

    # Policy registry
    policy_registry = {
        "mpc": MlpMpcPolicy,
        "mlp": MlpOnlyPolicy,
    }

    if policy_mode not in policy_registry:
        raise ValueError(f"Unsupported POLICY_MODE: {policy_mode}. Choose from {list(policy_registry.keys())}")

    # Create environment and policy
    env = RealDroneLineTrackEnv(device=device)
    policy_class = policy_registry[policy_mode]

    # Create PPO model
    model = PPO(
        policy=policy_class,
        env=env,
        verbose=0,
        n_steps=1024,
        batch_size=128,
        learning_rate=3e-4,
        gamma=0.99,
        clip_range=0.2,
    )

    # Train
    print(f"Starting training for {total_timesteps} timesteps...\n")
    model.learn(total_timesteps=total_timesteps, callback=EpisodeStatsCallback())

    # Save model with timestamp
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    current_time = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    model_name = f"ppo_{policy_mode}_{current_time}"
    model_path = output_dir / model_name

    # Create subdirectory for this run
    run_dir = output_dir / model_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save model and metadata
    model.save(run_dir / "policy")
    
    metadata = {
        "policy_mode": policy_mode,
        "total_timesteps": total_timesteps,
        "timestamp": current_time,
        "max_episode_steps": env.max_steps,
        "target_distance": env.target_distance,
        "target_speed": env.target_speed,
        "device": device,
    }
    
    import json
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Training Complete")
    print(f"{'='*80}")
    print(f"Model saved to: {run_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
