# Real Drone Line Tracking Control: Training & Validation

This package contains scripts for training and validating a PPO-based controller for drone line-tracking tasks using real quadrotor dynamics.

## Overview

### Task Definition
- **Objective**: Navigate from origin (0, 0, 0) to target position (-5, 0, 0)
- **Reference Motion**: Straight-line T-profile trajectory with start/end hover states
- **Duration**: 5 seconds nominal (250 steps at dt=0.02s)
- **Environment**: Real quadrotor dynamics from `drone.DroneDx`

### State & Control
- **State (10-dim)**: `x = [p(3), q(4), v(3)]`
  - Position, quaternion (attitude), velocity
  
- **Control (4-dim)**: `u = [f_c, ωx, ωy, ωz]`
  - Total thrust (N) and body-frame angular velocities (rad/s)

### Policy Modes
- **`mpc`**: Model Predictive Control with learned cost parameters
  - Requires environment variable: `ACMPC_T` (prediction horizon, default: 5)
  
- **`mlp`**: Pure neural network (multi-layer perceptron)
  - Direct mapping from state to control action

---

## Usage

### 1. Training

#### Train with MPC Policy (default)
```bash
cd AC-MPC\line_track
set POLICY_MODE=mpc
set ACMPC_T=5
set TOTAL_TIMESTEPS=100000
python train.py
```

#### Train with MLP Policy
```bash
set POLICY_MODE=mlp
set TOTAL_TIMESTEPS=100000
python train.py
```

#### Using Conda environment
```bash
conda activate AC-MPC
cd AC-MPC\line_track
set POLICY_MODE=mpc && python train.py
```

### Training Output
Models are saved to `output/ppo_{policy_mode}_{timestamp}/`:
```
output/
├── ppo_mpc_25-03-31_14-30-45/
│   ├── policy.zip          # Trained PPO model
│   └── metadata.json       # Training configuration
└── ppo_mlp_25-03-31_14-25-22/
    ├── policy.zip
    └── metadata.json
```

---

### 2. Validation

#### Run validation on trained model
```bash
python val.py --model_variant best --n_episodes 2 --model_dir output/ppo_mpc_04241231
```

#### Validation output options
```bash
# Save plots and data
python val.py --model_dir <model_dir> --n_episodes 5 --save_plots --save_data

# Deterministic evaluation only (no policy noise)
python val.py --model_dir <model_dir> --n_episodes 3 --deterministic
```

### Validation Outputs
Results are saved to `output/ppo_*/validation/`:
```
validation/
├── episode_000_trajectory.png  # 3×3 plot grid with trajectory, errors, controls
├── episode_001_trajectory.png
├── episode_000_data.npz        # Raw trajectory data (numpy format)
├── episode_001_data.npz
└── validation_summary.json     # Aggregate statistics
```

---

## Output Interpretation

### Plot Grid (3×3)
1. **3D Trajectory** (top-left)
   - Blue line: actual path
   - Red dashed: reference path
   - Green marker: start position
   - Red X: final actual position

2. **Position Error** (top-right)
   - Euclidean distance between actual and reference position
   - Includes mean and max values

3. **Position Components** (middle row)
   - X, Y, Z positions tracked separately
   - Blue: actual, Red dashed: reference

4. **X Velocity** (bottom-left)
   - Tracks velocity along primary axis
   - Reference comes from the planned T-profile trajectory

5. **Thrust Command** (bottom-middle)
   - Total thrust over time
   - Shows mean value and range

6. **Angular Velocities** (bottom-right)
   - Body-frame roll, pitch, yaw rates
   - Red, green, blue lines for wx, wy, wz

### Summary Statistics
Key metrics printed to console:
- **Position Error**: Mean, max, final (in meters)
- **Velocity Error**: Mean, max, final (in m/s)
- **Control Effort**: Mean/max thrust and angular velocities
- **Reward**: Total accumulated, average per step

---

## Key Design Decisions

### Real Dynamics Integration
- Uses `drone.DroneDx()` for forward propagation
- Supports batched state updates via PyTorch
- GPU acceleration available (auto-detected)

### Reward Function
```python
reward = 5.0                  # Base reward
   - 3.0 * pos_error      # Position tracking only
   - 0.01 * control_cost  # Minimize actuation
```

### MPC Integration
- For `mpc` mode, policy outputs learned cost parameters (Q, p)
- iLQR solver minimizes predicted trajectory cost
- Horizon length set via `ACMPC_T` environment variable

### Environment Structure
- Episode length: 250 steps (5 seconds)
- Termination: Out-of-bounds (>20m from origin)
- Truncation: Max steps reached
- No random disturbances (deterministic environment)

---

## Recommended Training Parameters

| Hyperparameter | Value | Notes |
|---|---|---|
| Total timesteps | 100,000 | Start with 50k for quick testing |
| n_steps (rollout) | 1024 | Per batch for gradient estimation |
| batch_size | 128 | Larger batches smoother gradients |
| learning_rate | 3e-4 | Tune down if diverging |
| gamma | 0.99 | Discount factor (5s episode) |
| clip_range | 0.2 | PPO clipping range |

---

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `train.py`
- Use CPU: `python train.py` (will auto-detect)

### MPC Solver Errors
- Check `ACMPC_T` (try 3 or 10 instead of 5)
- Verify `drone.DroneDx()` is properly initialized

### Poor Convergence
- Increase `total_timesteps`
- Lower `learning_rate` and increase `n_steps`
- Check reward function coefficients

---

## File Structure
```
line_track/
├── train.py              # Training script
├── val.py                # Validation script  
├── README.md             # This file
└── output/               # Generated during training
    ├── ppo_mpc_*/
    │   ├── policy.zip
    │   ├── metadata.json
    │   └── validation/
    │       ├── episode_*.png
    │       ├── episode_*.npz
    │       └── validation_summary.json
    └── ppo_mlp_*/
```

---

## Extensions

### Custom Flight Tasks
Modify `RealDroneLineTrackEnv` in `train.py`:
- `plan_straight_t_profile_trajectory()`: Change reference trajectory shape or timing law
- `step()`: Add obstacles, wind, sensor noise
- Reward function: Adjust coefficients for different priorities

### Policy Evaluation Metrics
Extend `TrajectoryAnalyzer` in `val.py`:
- Add stability margins
- Compute tracking indices (e.g., integral absolute error)
- Energy consumption analysis
- Compare multiple episodes/models

---

## Citation
If you use this code, please cite:
- AC-MPC framework: [your citation]
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
