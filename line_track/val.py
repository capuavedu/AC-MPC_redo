"""
Validation and analysis script for trained drone line-tracking controller.

This script loads a trained model and runs validation flight(s), analyzing:
- Control inputs (thrust, angular velocities)
- Actual trajectory vs reference trajectory
- Position, velocity, and attitude tracking errors
- Overall control effectiveness metrics
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional

# Add parent directory to path so we can import acmpc_public
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from stable_baselines3 import PPO

# Import environment
from train import RealDroneLineTrackEnv


class TrajectoryAnalyzer:
    """Analyzes and visualizes drone flight trajectory and control performance."""

    def __init__(self, save_dir: Optional[Path] = None):
        self.save_dir = save_dir
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for episode data
        self.trajectories = []

    def run_validation_episode(
        self,
        model: PPO,
        env: RealDroneLineTrackEnv,
        episode_idx: int = 0,
        deterministic: bool = True,
    ) -> Dict:
        """
        Run a single validation episode and collect detailed trajectory data.
        
        Returns:
            Dictionary with episode statistics and trajectories
        """
        obs, info = env.reset()
        done = False
        step = 0

        # Pre-allocate arrays for trajectory
        max_steps = env.max_steps
        traj = {
            "t": np.zeros(max_steps),
            "pos": np.zeros((max_steps, 3)),
            "pos_ref": np.zeros((max_steps, 3)),
            "vel": np.zeros((max_steps, 3)),
            "vel_ref": np.zeros((max_steps, 3)),
            "quat": np.zeros((max_steps, 4)),
            # Policy action before env-side remapping:
            # [normalized thrust delta, wx, wy, wz]
            "action_policy": np.zeros((max_steps, 4)),
            # Physical action sent into dynamics by env:
            # [f_c, wx, wy, wz]
            "action_physical": np.zeros((max_steps, 4)),
            "pos_error": np.zeros(max_steps),
            "vel_error": np.zeros(max_steps),
            "reward": np.zeros(max_steps),
        }

        while not done and step < max_steps:
            # Get action from policy
            action, _states = model.predict(obs, deterministic=deterministic)
            ref = env.get_reference(step)

            # Record pre-step state
            traj["t"][step] = step * env.dt
            traj["pos"][step] = obs[:3]
            traj["quat"][step] = obs[3:7]
            traj["vel"][step] = obs[7:10]
            traj["pos_ref"][step] = ref["position"]
            traj["vel_ref"][step] = ref["velocity"]
            traj["action_policy"][step] = action

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record error metrics
            traj["pos_error"][step] = info.get("pos_error", 0.0)
            traj["vel_error"][step] = info.get("vel_error", 0.0)
            traj["action_physical"][step] = info.get("action_physical", np.zeros(4))
            traj["reward"][step] = float(reward)

            step += 1

        # Trim arrays to actual steps
        for key in traj:
            traj[key] = traj[key][:step]

        # Compute statistics
        stats = {
            "episode": episode_idx,
            "n_steps": step,
            "duration": step * env.dt,
            "final_pos": traj["pos"][-1].copy(),
            "final_vel": traj["vel"][-1].copy(),
            "final_pos_error": traj["pos_error"][-1],
            "final_vel_error": traj["vel_error"][-1],
            "pos_error_mean": float(np.mean(traj["pos_error"])),
            "pos_error_max": float(np.max(traj["pos_error"])),
            "vel_error_mean": float(np.mean(traj["vel_error"])),
            "vel_error_max": float(np.max(traj["vel_error"])),
            "total_reward": float(np.sum(traj["reward"])),
            "avg_reward": float(np.mean(traj["reward"])),
        }

        # Thrust statistics
        thrust = traj["action_physical"][:, 0]
        stats["thrust_mean"] = float(np.mean(thrust))
        stats["thrust_max"] = float(np.max(thrust))
        stats["thrust_min"] = float(np.min(thrust))

        # Angular velocity statistics
        for i, name in enumerate(["wx", "wy", "wz"]):
            omega = traj["action_physical"][:, i + 1]
            stats[f"{name}_mean"] = float(np.mean(np.abs(omega)))
            stats[f"{name}_max"] = float(np.max(np.abs(omega)))

        self.trajectories.append({"stats": stats, "traj": traj})
        return {"stats": stats, "traj": traj}

    def plot_trajectory(self, result: Dict, save_name: str = "trajectory.png"):
        """Generate comprehensive trajectory visualization."""
        stats = result["stats"]
        traj = result["traj"]

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

        # 1. 3D Trajectory
        ax = fig.add_subplot(gs[0, :2], projection="3d")
        ax.plot(traj["pos"][:, 0], traj["pos"][:, 1], traj["pos"][:, 2], 
                "b-", linewidth=2, label="Actual")
        ax.plot(traj["pos_ref"][:, 0], traj["pos_ref"][:, 1], traj["pos_ref"][:, 2], 
                "r--", linewidth=2, label="Reference")
        ax.scatter([0], [0], [0], c="g", s=100, marker="o", label="Start")
        ax.scatter([traj["pos"][-1, 0]], [traj["pos"][-1, 1]], [traj["pos"][-1, 2]], 
                   c="r", s=100, marker="x", label="End (actual)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D Trajectory")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Position Error (XYZ)
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(traj["t"], traj["pos_error"], "r-", linewidth=2)
        ax.fill_between(traj["t"], 0, traj["pos_error"], alpha=0.3)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title(f"Position Tracking Error\nMean: {stats['pos_error_mean']:.4f}m, Max: {stats['pos_error_max']:.4f}m")
        ax.grid(True, alpha=0.3)

        # 3. Position X, Y, Z separately
        ax = fig.add_subplot(gs[1, 0])
        ax.plot(traj["t"], traj["pos"][:, 0], "b-", linewidth=2, label="Actual")
        ax.plot(traj["t"], traj["pos_ref"][:, 0], "r--", linewidth=2, label="Reference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("X (m)")
        ax.set_title("X Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        ax.plot(traj["t"], traj["pos"][:, 1], "b-", linewidth=2, label="Actual")
        ax.plot(traj["t"], traj["pos_ref"][:, 1], "r--", linewidth=2, label="Reference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Y Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 2])
        ax.plot(traj["t"], traj["pos"][:, 2], "b-", linewidth=2, label="Actual")
        ax.plot(traj["t"], traj["pos_ref"][:, 2], "r--", linewidth=2, label="Reference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Z (m)")
        ax.set_title("Z Position")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Velocity
        ax = fig.add_subplot(gs[2, 0])
        ax.plot(traj["t"], traj["vel"][:, 0], "b-", linewidth=2, label="Actual")
        ax.plot(traj["t"], traj["vel_ref"][:, 0], "r--", linewidth=2, label="Reference")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Vx (m/s)")
        ax.set_title("X Velocity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[2, 1])
        ax.plot(traj["t"], traj["action_physical"][:, 0], "g-", linewidth=2)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control (N)")
        ax.set_title(f"Thrust Command\nMean: {stats['thrust_mean']:.2f}N, Max: {stats['thrust_max']:.2f}N")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=stats["thrust_mean"], color="orange", linestyle=":", label="Mean")
        ax.legend()

        # 5. Angular velocities
        ax = fig.add_subplot(gs[2, 2])
        ax.plot(traj["t"], traj["action_physical"][:, 1], "r-", linewidth=1.5, label="wx")
        ax.plot(traj["t"], traj["action_physical"][:, 2], "g-", linewidth=1.5, label="wy")
        ax.plot(traj["t"], traj["action_physical"][:, 3], "b-", linewidth=1.5, label="wz")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Velocity (rad/s)")
        ax.set_title("Body Angular Velocities")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Episode {stats['episode']}: {stats['n_steps']} steps, "
            f"Final Pos Error: {stats['final_pos_error']:.4f}m",
            fontsize=14,
            fontweight="bold"
        )

        if self.save_dir:
            path = self.save_dir / save_name
            fig.savefig(str(path), dpi=150, bbox_inches="tight")
            print(f"✓ Saved trajectory plot to: {path}")

        plt.close(fig)

    def print_episode_summary(self, result: Dict):
        """Print detailed episode statistics."""
        stats = result["stats"]
        
        print(f"\n{'='*80}")
        print(f"EPISODE {stats['episode']:03d} SUMMARY")
        print(f"{'='*80}")
        print(f"Duration:           {stats['duration']:.2f}s ({stats['n_steps']} steps)")
        print(f"\nPosition Tracking:")
        print(f"  Final position:    {stats['final_pos']}")
        print(f"  Final error:       {stats['final_pos_error']:.4f}m")
        print(f"  Mean error:        {stats['pos_error_mean']:.4f}m")
        print(f"  Max error:         {stats['pos_error_max']:.4f}m")
        print(f"\nVelocity Tracking:")
        print(f"  Final velocity:    {stats['final_vel']}")
        print(f"  Final error:       {stats['final_vel_error']:.4f}m/s")
        print(f"  Mean error:        {stats['vel_error_mean']:.4f}m/s")
        print(f"  Max error:         {stats['vel_error_max']:.4f}m/s")
        print(f"\nControl Effort:")
        print(f"  Thrust (mean/max): {stats['thrust_mean']:.2f}N / {stats['thrust_max']:.2f}N")
        print(f"  Wx (mean/max):     {stats['wx_mean']:.4f} / {stats['wx_max']:.4f} rad/s")
        print(f"  Wy (mean/max):     {stats['wy_mean']:.4f} / {stats['wy_max']:.4f} rad/s")
        print(f"  Wz (mean/max):     {stats['wz_mean']:.4f} / {stats['wz_max']:.4f} rad/s")
        print(f"\nRewards:")
        print(f"  Total:             {stats['total_reward']:.2f}")
        print(f"  Average per step:  {stats['avg_reward']:.4f}")
        print(f"{'='*80}\n")

    def save_detailed_data(self, result: Dict, save_name: str = "trajectory_data.npz"):
        """Save detailed trajectory data for further analysis."""
        if not self.save_dir:
            return

        np.savez(
            self.save_dir / save_name,
            **result["traj"],
            **{"stats_" + k: v for k, v in result["stats"].items()}
        )
        print(f"✓ Saved trajectory data to: {self.save_dir / save_name}")


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


def _setup_validation_logging(results_dir: Path):
    """Redirect stdout/stderr to both terminal and validation-local log file."""
    log_path = results_dir / "validation_log.txt"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)
    return log_file, original_stdout, original_stderr, log_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate trained drone controller")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to trained model directory (output of train.py)",
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=3,
        help="Number of validation episodes to run",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy (no exploration noise)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        default=True,
        help="Save trajectory plots",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        default=True,
        help="Save detailed trajectory data",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="final",
        choices=["final", "best"],
        help="Which checkpoint to validate: final=policy, best=policy_best",
    )

    args = parser.parse_args()

    # Load model and metadata
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"✗ Model directory not found: {model_dir}")
        sys.exit(1)

    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"✗ Metadata file not found: {metadata_path}")
        sys.exit(1)

    # Create output directory for validation results (split by final/best checkpoint).
    results_suffix = "validation" if args.model_variant == "final" else "validation_best"
    results_dir = model_dir / results_suffix
    results_dir.mkdir(parents=True, exist_ok=True)

    log_file = None
    original_stdout = None
    original_stderr = None
    log_path = None

    try:
        log_file, original_stdout, original_stderr, log_path = _setup_validation_logging(results_dir)

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"\n{'='*80}")
        print(f"VALIDATION CONFIGURATION")
        print(f"{'='*80}")
        print(f"Model directory:    {model_dir}")
        print(f"Policy mode:        {metadata['policy_mode']}")
        print(f"Training timesteps: {metadata['total_timesteps']}")
        print(f"Validation episodes: {args.n_episodes}")
        print(f"Model variant:      {args.model_variant}")
        print(f"Log Path:           {log_path}")
        #print(f"Device:             {metadata['device']}")
        print(f"{'='*80}\n")

        # MPC policy class expects ACMPC_T to be present (set during training).
        if metadata.get("policy_mode", "") == "mpc":
            os.environ.setdefault("ACMPC_T", "2")

        # Load model from stable-baselines3 archive.
        saved_models = metadata.get("saved_models", {})
        default_name = "policy.zip" if args.model_variant == "final" else "policy_best.zip"
        model_file_name = saved_models.get(args.model_variant, default_name)
        model_path = model_dir / model_file_name

        if not model_path.exists():
            fallback_stem = "policy" if args.model_variant == "final" else "policy_best"
            model_path = model_dir / fallback_stem

        try:
            model = PPO.load(model_path)
            print(f"✓ Loaded model from: {model_path}\n")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            if isinstance(e, KeyError) and "ACMPC_T" in str(e):
                print("Hint: set ACMPC_T before validation, e.g. set ACMPC_T=2")
            sys.exit(1)

        # Set up environment (match training metadata to avoid train/val task mismatch)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_episode_steps = int(metadata.get("max_episode_steps", 250))
        target_distance = float(metadata.get("target_distance", 5.0))
        start_state = metadata.get("start_state")
        goal_state = metadata.get("goal_state")
        start_state_np = np.asarray(start_state, dtype=np.float32).reshape(10) if start_state is not None else None
        goal_state_np = np.asarray(goal_state, dtype=np.float32).reshape(10) if goal_state is not None else None

        env = RealDroneLineTrackEnv(
            max_steps=max_episode_steps,
            target_distance=target_distance,
            start_state=start_state_np,
            goal_state=goal_state_np,
            device=str(device),
        )

        print(
            "Validation env: "
            f"max_steps={env.max_steps}, "
            f"target_distance={env.target_distance:.3f}, "
            f"dt={env.dt:.3f}s"
        )

        analyzer = TrajectoryAnalyzer(save_dir=results_dir if (args.save_plots or args.save_data) else None)

        # Run validation episodes
        print(f"Running {args.n_episodes} validation episode(s)...\n")
        for ep in range(args.n_episodes):
            result = analyzer.run_validation_episode(
                model=model,
                env=env,
                episode_idx=ep,
                deterministic=args.deterministic,
            )
            analyzer.print_episode_summary(result)
            
            if args.save_plots:
                analyzer.plot_trajectory(result, save_name=f"episode_{ep:03d}_trajectory.png")
            if args.save_data:
                analyzer.save_detailed_data(result, save_name=f"episode_{ep:03d}_data.npz")

        # Save summary statistics
        if analyzer.trajectories:
            # Convert numpy arrays to native Python types for JSON serialization
            def convert_to_native(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(v) for v in obj]
                return obj

            summary = {
                "n_episodes": len(analyzer.trajectories),
                "episodes": [convert_to_native(t["stats"]) for t in analyzer.trajectories],
            }
            
            # Compute aggregate statistics
            all_pos_errors = np.concatenate([t["traj"]["pos_error"] for t in analyzer.trajectories])
            all_vel_errors = np.concatenate([t["traj"]["vel_error"] for t in analyzer.trajectories])
            
            summary["aggregate"] = {
                "pos_error_mean": float(np.mean(all_pos_errors)),
                "pos_error_std": float(np.std(all_pos_errors)),
                "vel_error_mean": float(np.mean(all_vel_errors)),
                "vel_error_std": float(np.std(all_vel_errors)),
            }

            if results_dir:
                with open(results_dir / "validation_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                print(f"\n✓ Saved validation summary to: {results_dir / 'validation_summary.json'}")

        print(f"\n{'='*80}")
        print(f"VALIDATION COMPLETE")
        print(f"Results saved to: {results_dir}")
        print(f"{'='*80}\n")
    finally:
        if log_file is not None:
            try:
                sys.stdout = original_stdout
                sys.stderr = original_stderr
            finally:
                log_file.close()


if __name__ == "__main__":
    main()
