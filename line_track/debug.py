"""
Debug tool for trajectory/control consistency with DroneDx dynamics.

What this script does:
1. Build a T-profile reference trajectory from start/goal states.
2. Compute desired 3-axis acceleration by finite difference on reference velocity.
3. Compute desired body angular velocity from quaternion increments.
4. Back-solve expected control u=[f_c, wx, wy, wz] under DroneDx CTBR model.
5. Run one-step DroneDx check and report acceleration mismatch.

Usage examples:
  python line_track/debug.py
  python line_track/debug.py --max_steps 1000 --dt 0.02
  python line_track/debug.py --metadata line_track/output/ppo_mlp_26-04-02_01-00-59/metadata.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch

# Add repo root to import path
sys.path.insert(0, str(Path(__file__).parent.parent))

from train import plan_straight_t_profile_trajectory  # noqa: E402
from acmpc_public.diff_mpc_drones import drone  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_conj(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quat_to_body_z_world(q: np.ndarray) -> np.ndarray:
    """Return world-frame body z axis b3 from quaternion [w,x,y,z]."""
    q = _normalize_quat(q)
    w, x, y, z = q
    # Third column of rotation matrix R(q)
    return np.array(
        [2.0 * (x * z + w * y), 2.0 * (y * z - w * x), w * w - x * x - y * y + z * z],
        dtype=np.float64,
    )


def _quat_to_yaw(q: np.ndarray) -> float:
    """Extract world yaw angle from quaternion [w,x,y,z]."""
    q = _normalize_quat(q)
    w, x, y, z = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    return _normalize_quat(np.array([qw, qx, qy, qz], dtype=np.float64))


def _quat_from_b3_and_yaw(b3_world: np.ndarray, yaw: float) -> np.ndarray:
    """Construct quaternion with body-z aligned to b3_world and yaw preference."""
    b3 = np.asarray(b3_world, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(b3))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    b3 = b3 / n

    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float64)
    b2 = np.cross(b3, x_c)
    if np.linalg.norm(b2) < 1e-9:
        x_c = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        b2 = np.cross(b3, x_c)
    if np.linalg.norm(b2) < 1e-9:
        x_c = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        b2 = np.cross(b3, x_c)

    b2 = b2 / max(np.linalg.norm(b2), 1e-12)
    b1 = np.cross(b2, b3)
    b1 = b1 / max(np.linalg.norm(b1), 1e-12)

    R = np.column_stack([b1, b2, b3])
    return _rotmat_to_quat(R)


def _finite_diff_velocity(v: np.ndarray, dt: float) -> np.ndarray:
    """Central difference inside, one-sided at boundaries."""
    a = np.zeros_like(v, dtype=np.float64)
    if len(v) < 2:
        return a
    a[0] = (v[1] - v[0]) / dt
    a[-1] = (v[-1] - v[-2]) / dt
    if len(v) > 2:
        a[1:-1] = (v[2:] - v[:-2]) / (2.0 * dt)
    return a


def _compute_omega_from_quat(quat: np.ndarray, dt: float) -> np.ndarray:
    """Compute body angular velocity from quaternion sequence."""
    n = quat.shape[0]
    omega = np.zeros((n, 3), dtype=np.float64)
    if n < 2:
        return omega

    for k in range(n - 1):
        qk = _normalize_quat(quat[k])
        qk1 = _normalize_quat(quat[k + 1])
        dq = _quat_mul(_quat_conj(qk), qk1)
        dq = _normalize_quat(dq)

        # Ensure shortest arc
        if dq[0] < 0.0:
            dq = -dq

        w = float(np.clip(dq[0], -1.0, 1.0))
        angle = 2.0 * np.arccos(w)
        s = np.sqrt(max(1e-12, 1.0 - w * w))
        axis = dq[1:4] / s if s > 1e-6 else np.array([0.0, 0.0, 0.0], dtype=np.float64)
        omega[k] = axis * angle / dt

    omega[-1] = omega[-2]
    return omega


def _torch_quat_yaw(q: torch.Tensor) -> torch.Tensor:
    """Extract yaw from quaternion tensor [w,x,y,z]."""
    siny_cosp = 2.0 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    return torch.atan2(siny_cosp, cosy_cosp)


def _quat_alignment_loss(q_pred: torch.Tensor, q_tgt: torch.Tensor) -> torch.Tensor:
    """Quaternion distance invariant to sign (q and -q represent same attitude)."""
    loss_pos = torch.sum((q_pred - q_tgt) ** 2)
    loss_neg = torch.sum((q_pred + q_tgt) ** 2)
    return torch.minimum(loss_pos, loss_neg)


def _solve_attitude_and_thrust_with_drone_interface(
    dx: drone.DroneDx,
    pos_k: np.ndarray,
    vel_k: np.ndarray,
    acc_des_k: np.ndarray,
    yaw_ref_k: float,
    q_init: np.ndarray,
    iters: int = 80,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Solve q,f_c by minimizing acceleration mismatch via DroneDx.nonlinear_dynamics.
    This directly validates DroneDx translational interface.
    """
    thrust_min = float(dx.thrust_min)
    thrust_max = float(dx.thrust_max)

    pos_t = torch.as_tensor(pos_k, dtype=torch.float32)
    vel_t = torch.as_tensor(vel_k, dtype=torch.float32)
    acc_des_t = torch.as_tensor(acc_des_k, dtype=torch.float32)
    q_init_t = torch.as_tensor(_normalize_quat(q_init), dtype=torch.float32)

    raw_q = torch.nn.Parameter(q_init_t.clone())
    g_norm = float(torch.linalg.norm(dx.g).cpu().item())
    mass = float(dx.mass)
    fc_guess = np.clip(mass * g_norm, thrust_min, thrust_max)
    raw_fc = torch.nn.Parameter(torch.tensor(fc_guess, dtype=torch.float32))
    opt = torch.optim.Adam([raw_q, raw_fc], lr=0.05)

    for _ in range(iters):
        opt.zero_grad()
        q = raw_q / torch.clamp(torch.linalg.norm(raw_q), min=1e-8)
        fc = torch.clamp(raw_fc, min=thrust_min, max=thrust_max)

        xk = torch.cat([pos_t, q, vel_t], dim=0)
        uk = torch.stack([fc, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)])
        xdot = dx.nonlinear_dynamics(xk, uk)
        acc_model = xdot[7:10]

        loss_acc = torch.mean((acc_model - acc_des_t) ** 2)
        yaw_cur = _torch_quat_yaw(q)
        yaw_ref_t = torch.tensor(yaw_ref_k, dtype=torch.float32)
        loss_yaw = (yaw_cur - yaw_ref_t) ** 2
        # Keep attitude close to initialization to avoid unnecessary spin around b3.
        loss_smooth = _quat_alignment_loss(q, q_init_t)
        loss = loss_acc + 5e-3 * loss_yaw + 2e-3 * loss_smooth
        loss.backward()
        opt.step()

    with torch.no_grad():
        q = raw_q / torch.clamp(torch.linalg.norm(raw_q), min=1e-8)
        fc = torch.clamp(raw_fc, min=thrust_min, max=thrust_max)
        xk = torch.cat([pos_t, q, vel_t], dim=0)
        uk = torch.stack([fc, torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)])
        xdot = dx.nonlinear_dynamics(xk, uk)
        acc_model = xdot[7:10]

    return (
        q.cpu().numpy().astype(np.float64),
        float(fc.cpu().item()),
        acc_model.cpu().numpy().astype(np.float64),
    )


def _solve_omega_with_drone_forward(
    dx: drone.DroneDx,
    state_k: np.ndarray,
    fc_k: float,
    q_next_target: np.ndarray,
    iters: int = 60,
) -> np.ndarray:
    """Solve omega so DroneDx.forward quaternion matches target next quaternion."""
    xk = torch.as_tensor(state_k.astype(np.float32), dtype=torch.float32)
    q_tgt = torch.as_tensor(_normalize_quat(q_next_target).astype(np.float32), dtype=torch.float32)

    raw_w = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32))
    opt = torch.optim.Adam([raw_w], lr=0.06)
    w_lim = dx.omega_max.to(dtype=torch.float32)

    for _ in range(iters):
        opt.zero_grad()
        w = torch.clamp(raw_w, min=-w_lim, max=w_lim)
        uk = torch.cat([torch.tensor([fc_k], dtype=torch.float32), w], dim=0)
        xk1 = dx.forward(xk.unsqueeze(0), uk.unsqueeze(0)).squeeze(0)
        q_pred = xk1[3:7]
        loss = _quat_alignment_loss(q_pred, q_tgt)
        loss.backward()
        opt.step()

    with torch.no_grad():
        w = torch.clamp(raw_w, min=-w_lim, max=w_lim)
    return w.cpu().numpy().astype(np.float64)


def compute_expected_control(
    states: np.ndarray,
    dt: float,
    dx: drone.DroneDx,
) -> Dict[str, np.ndarray]:
    """
    Back-solve expected control from reference trajectory under DroneDx CTBR model.

        Solve expected control/state by directly calling DroneDx interfaces.
        - q,f_c are solved from DroneDx.nonlinear_dynamics acceleration matching.
        - omega is solved from DroneDx.forward quaternion matching.
    """
    pos = states[:, 0:3].astype(np.float64)
    quat_ref = states[:, 3:7].astype(np.float64)
    vel = states[:, 7:10].astype(np.float64)

    g_world = dx.g.cpu().numpy().astype(np.float64)
    acc = _finite_diff_velocity(vel, dt)

    n = states.shape[0]
    quat_dyn = np.zeros((n, 4), dtype=np.float64)
    fc_raw = np.zeros(n, dtype=np.float64)
    fc_clipped = np.zeros(n, dtype=np.float64)
    accel_model = np.zeros((n, 3), dtype=np.float64)
    accel_error = np.zeros((n, 3), dtype=np.float64)

    yaw_ref = np.array([_quat_to_yaw(q) for q in quat_ref], dtype=np.float64)
    q_prev = quat_ref[0]

    for k in range(n):
        q_sol, fc_sol, a_model = _solve_attitude_and_thrust_with_drone_interface(
            dx=dx,
            pos_k=pos[k],
            vel_k=vel[k],
            acc_des_k=acc[k],
            yaw_ref_k=yaw_ref[k],
            q_init=q_prev,
        )
        quat_dyn[k] = q_sol
        fc_clipped[k] = fc_sol
        fc_raw[k] = fc_sol
        accel_model[k] = a_model
        accel_error[k] = acc[k] - accel_model[k]
        q_prev = q_sol

    omega = np.zeros((n, 3), dtype=np.float64)
    for k in range(n - 1):
        state_k = np.concatenate([pos[k], quat_dyn[k], vel[k]], axis=0)
        omega[k] = _solve_omega_with_drone_forward(
            dx=dx,
            state_k=state_k,
            fc_k=fc_clipped[k],
            q_next_target=quat_dyn[k + 1],
        )
    if n >= 2:
        omega[-1] = omega[-2]

    u_expected = np.column_stack([fc_clipped, omega])
    states_dyn = np.concatenate([pos, quat_dyn, vel], axis=1)

    return {
        "position": pos,
        "velocity": vel,
        "quaternion": quat_dyn,
        "quaternion_ref": quat_ref,
        "acc_world": acc,
        "omega_body": omega,
        "fc_expected": fc_clipped,
        "fc_unclipped": fc_raw,
        "u_expected": u_expected,
        "state_dyn": states_dyn,
        "acc_parallel_world": accel_model,
        "acc_residual_world": accel_error,
        "acc_residual_norm": np.linalg.norm(accel_error, axis=1),
    }


def rollout_one_step_check(
    dx: drone.DroneDx,
    states: np.ndarray,
    u_expected: np.ndarray,
    dt: float,
) -> Dict[str, np.ndarray]:
    """Compare finite-difference acceleration to one-step DroneDx acceleration."""
    n = states.shape[0]
    a_fd = np.zeros((n, 3), dtype=np.float64)
    a_model = np.zeros((n, 3), dtype=np.float64)

    if n < 2:
        return {
            "acc_fd": a_fd,
            "acc_model": a_model,
            "acc_model_minus_fd": a_model - a_fd,
            "acc_model_minus_fd_norm": np.linalg.norm(a_model - a_fd, axis=1),
        }

    vel = states[:, 7:10].astype(np.float64)
    a_fd[:-1] = (vel[1:] - vel[:-1]) / dt
    a_fd[-1] = a_fd[-2]

    with torch.no_grad():
        for k in range(n - 1):
            xk = torch.as_tensor(states[k], dtype=torch.float32)
            uk = torch.as_tensor(u_expected[k], dtype=torch.float32)
            xk1 = dx.forward(xk.unsqueeze(0), uk.unsqueeze(0)).squeeze(0)
            vk = xk[7:10].cpu().numpy().astype(np.float64)
            vk1 = xk1[7:10].cpu().numpy().astype(np.float64)
            a_model[k] = (vk1 - vk) / dt
        a_model[-1] = a_model[-2]

    diff = a_model - a_fd
    return {
        "acc_fd": a_fd,
        "acc_model": a_model,
        "acc_model_minus_fd": diff,
        "acc_model_minus_fd_norm": np.linalg.norm(diff, axis=1),
    }


def _default_states(target_distance: float) -> Tuple[np.ndarray, np.ndarray]:
    start = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([-target_distance, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return start, goal


def _print_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    print(
        f"{name:28s} min={arr.min(): .6f} max={arr.max(): .6f} "
        f"mean={arr.mean(): .6f} std={arr.std(): .6f}"
    )


def save_debug_plots(
    save_dir: Path,
    time: np.ndarray,
    expected: Dict[str, np.ndarray],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)

    t = np.asarray(time, dtype=np.float64)
    pos = expected["position"]
    vel = expected["velocity"]
    speed = np.linalg.norm(vel, axis=1)
    omega = expected["omega_body"]
    acc = expected["acc_world"]
    u = expected["u_expected"]

    # 1) Trajectory
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=2.0)
    ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], s=40, label="start")
    ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], s=40, label="goal")
    ax.set_title("Trajectory (3D)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "trajectory_3d.png", dpi=180)
    plt.close(fig)

    # 2) Desired speed
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, speed, linewidth=2.0, label="|v_des|")
    ax.plot(t, vel[:, 0], linewidth=1.2, label="v_des_x")
    ax.plot(t, vel[:, 1], linewidth=1.2, label="v_des_y")
    ax.plot(t, vel[:, 2], linewidth=1.2, label="v_des_z")
    ax.set_title("Desired Velocity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("velocity [m/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "desired_velocity.png", dpi=180)
    plt.close(fig)

    # 3) Desired angular velocity
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, omega[:, 0], linewidth=1.6, label="omega_des_x")
    ax.plot(t, omega[:, 1], linewidth=1.6, label="omega_des_y")
    ax.plot(t, omega[:, 2], linewidth=1.6, label="omega_des_z")
    ax.set_title("Desired Angular Velocity")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("angular velocity [rad/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "desired_angular_velocity.png", dpi=180)
    plt.close(fig)

    # 4) Desired acceleration
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, acc[:, 0], linewidth=1.6, label="a_des_x")
    ax.plot(t, acc[:, 1], linewidth=1.6, label="a_des_y")
    ax.plot(t, acc[:, 2], linewidth=1.6, label="a_des_z")
    ax.set_title("Desired Acceleration")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("acceleration [m/s^2]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "desired_acceleration.png", dpi=180)
    plt.close(fig)

    # 5) Desired control input u=[f_c, wx, wy, wz]
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axs = axs.ravel()
    labels = ["f_c [N]", "w_x [rad/s]", "w_y [rad/s]", "w_z [rad/s]"]
    for i in range(4):
        axs[i].plot(t, u[:, i], linewidth=1.6)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True, alpha=0.3)
    axs[2].set_xlabel("time [s]")
    axs[3].set_xlabel("time [s]")
    fig.suptitle("Desired Control Input")
    fig.tight_layout()
    fig.savefig(save_dir / "desired_control.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug T-profile trajectory against DroneDx dynamics")
    parser.add_argument("--metadata", type=str, default="", help="Optional metadata.json path")
    parser.add_argument("--max_steps", type=int, default=250, help="Trajectory steps")
    parser.add_argument("--dt", type=float, default=0.02, help="Step time")
    parser.add_argument("--target_distance", type=float, default=5.0, help="Goal distance in meters")
    parser.add_argument("--save_dir", type=str, default="line_track/output/debug_figure", help="Directory to save plots")
    args = parser.parse_args()

    start_state, goal_state = _default_states(args.target_distance)
    max_steps = int(args.max_steps)
    dt = float(args.dt)

    if args.metadata:
        metadata_path = Path(args.metadata)
        with open(metadata_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        max_steps = int(meta.get("max_episode_steps", max_steps))
        start_raw = meta.get("start_state")
        goal_raw = meta.get("goal_state")
        if start_raw is not None:
            start_state = np.asarray(start_raw, dtype=np.float32).reshape(10)
        if goal_raw is not None:
            goal_state = np.asarray(goal_raw, dtype=np.float32).reshape(10)
        print(f"Loaded metadata: {metadata_path}")

    print("=" * 88)
    print("Trajectory + Dynamics Debug")
    print("=" * 88)
    print(f"max_steps={max_steps}, dt={dt:.4f}s, total_time={max_steps * dt:.3f}s")
    print(f"start_state={start_state.tolist()}")
    print(f"goal_state ={goal_state.tolist()}")

    traj = plan_straight_t_profile_trajectory(
        start_state=start_state,
        goal_state=goal_state,
        max_steps=max_steps,
        dt=dt,
    )
    states = traj["state"].astype(np.float64)

    dx = drone.DroneDx(device="cpu")
    expected = compute_expected_control(
        states=states,
        dt=dt,
        dx=dx,
    )

    check = rollout_one_step_check(
        dx=dx,
        states=expected["state_dyn"],
        u_expected=expected["u_expected"],
        dt=dt,
    )

    _print_stats("acc_x (m/s^2)", expected["acc_world"][:, 0])
    _print_stats("acc_y (m/s^2)", expected["acc_world"][:, 1])
    _print_stats("acc_z (m/s^2)", expected["acc_world"][:, 2])
    _print_stats("omega_x (rad/s)", expected["omega_body"][:, 0])
    _print_stats("omega_y (rad/s)", expected["omega_body"][:, 1])
    _print_stats("omega_z (rad/s)", expected["omega_body"][:, 2])
    _print_stats("f_c_expected (N)", expected["fc_expected"])
    _print_stats("f_c_unclipped (N)", expected["fc_unclipped"])
    _print_stats("acc_residual_norm", expected["acc_residual_norm"])
    _print_stats("model_minus_fd_norm", check["acc_model_minus_fd_norm"])

    print("\nKey interpretation:")
    print("- This debug now reconstructs a dynamics-consistent attitude from desired acceleration and DroneDx CTBR.")
    print("- If f_c_unclipped exceeds DroneDx thrust limits, clipping will introduce acc_residual_norm.")

    save_dir = Path(args.save_dir)
    save_debug_plots(
        save_dir=save_dir,
        time=traj["time"],
        expected=expected,
    )
    print(f"\nSaved debug figures to: {save_dir}")
    print("- trajectory_3d.png")
    print("- desired_velocity.png")
    print("- desired_angular_velocity.png")
    print("- desired_acceleration.png")
    print("- desired_control.png")
    print("=" * 88)


if __name__ == "__main__":
    main()
