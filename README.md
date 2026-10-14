# Bottle-Dynamics

A small custom bottle-flip environment using Gymnasium, Pymunk, and Stable-Baselines3.

## Environment

- The bottle is a rigid 2D rectangle standing in for an **empty reusable plastic bottle**.
- Bottle mass is an initial estimate of **0.12 kg**, not a measured value.
- The bottle swings from a pivot; the agent controls torque and chooses when to release.
- After release, Pymunk simulates free flight and collisions with a **continuous ground surface**.
- The green strip marks the target zone on the ground; it is not a separate floating platform.
- Water/sloshing physics is disabled for this baseline.

A success requires an upright, low-speed, stable landing in the target zone and enough rotation. Merely touching the floor does not count.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

Each algorithm has its own folder and its own `train.py` and `evaluate.py`.

```powershell
python PPO/train.py --timesteps 500000 --checkpoint-every 10000
python A2C/train.py --timesteps 500000 --checkpoint-every 10000
python SAC/train.py --timesteps 500000 --checkpoint-every 10000
```

Each command performs **one continuous training run**. At each checkpoint interval the current model is saved while learning continues; training is not reset or split into separate runs. A 500,000-step run saved every 10,000 steps produces around 50 checkpoint files, plus `final_model.zip`.

Outputs are separated by algorithm. For example, PPO writes to `PPO/runs/seed42/`. Use `--seed 123` or `--run-name experiment2` to create another run. The trainer refuses to overwrite a non-empty run directory.

## TensorBoard

Open the run’s log folder, for example:

```powershell
tensorboard --logdir PPO/runs/seed42/tensorboard
```

Only these two scalar metrics are written:

- `rollout/ep_rew_mean` — mean reward over the last 100 completed episodes.
- `rollout/ep_len_mean` — mean number of agent decisions per episode over the last 100 completed episodes.

Checkpoint and evaluation details are stored in files, not as extra TensorBoard metrics.

## Evaluate and replay checkpoints

Evaluate every checkpoint on the same set of episode seeds:

```powershell
python PPO/evaluate.py --run-name seed42 --episodes 20
```

The script writes `evaluation_results.csv` and ranks candidates by success rate, then mean reward. Replay one chosen checkpoint:

```powershell
python PPO/evaluate.py --run-name seed42 --model PPO/runs/seed42/checkpoints/ppo_10000_steps.zip --episodes 1 --render
```

Use the matching folder for A2C or SAC. For example, `python SAC/evaluate.py --run-name seed42 --episodes 20`.

## Fair comparison

Train PPO, A2C, and SAC using the same environment, timestep budget, and seed before comparing them. Evaluate checkpoints on identical seeds. The older experiments used a different environment version, so their scores should be treated as historical baselines, not direct apples-to-apples comparisons.
