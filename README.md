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

## Files

Each algorithm folder has its own, self-contained `train.py` and `evaluate.py`, similar to a basic Stable-Baselines3 example. The scripts directly create or load the model, run episodes, print results, and write TensorBoard metrics. PPO, A2C, and SAC outputs are kept separate.

```text
PPO/
  train.py
  evaluate.py
  runs/
A2C/
  train.py
  evaluate.py
  runs/
SAC/
  train.py
  evaluate.py
  runs/
bottle_env.py
```

## Train

The default training budget is 500,000 timesteps, with a checkpoint saved every 10,000 steps and seed 42. Start with a short smoke test:

```powershell
python PPO/train.py --timesteps 2000 --checkpoint-every 1000 --run-name test_run
```

For longer runs:

```powershell
python PPO/train.py --timesteps 500000 --checkpoint-every 10000
python A2C/train.py --timesteps 500000 --checkpoint-every 10000
python SAC/train.py --timesteps 500000 --checkpoint-every 10000
```

Each command performs one continuous training run. Checkpoints are saved while learning continues, followed by `final_model.zip`. For example, PPO writes to `PPO/runs/seed42/`, including:

- `checkpoints/` — periodic model snapshots
- `tensorboard/` — Stable-Baselines3 training metrics
- `final_model.zip` — model saved at the end

Use `--seed 123` or `--run-name experiment2` to create a different run. A script refuses to overwrite a non-empty run folder; choose a new name for a new experiment.

## TensorBoard

For example, to view PPO's training metrics:

```powershell
tensorboard --logdir PPO/runs/seed42/tensorboard
```

Open the local URL printed by TensorBoard. For A2C or SAC, use that algorithm's matching folder.

## Evaluate

By default, evaluation loads the final model and runs five episodes, following the same basic loop as the standalone A2C evaluation example:

```powershell
python PPO/evaluate.py --run-name seed42 --episodes 5
```

Each episode prints its reward, length, termination status, success flag, and outcome. The script writes summary results to `evaluation_results.csv` and evaluation metrics under `evaluation_tensorboard/` within that run folder.

Evaluate all periodic checkpoints plus the final model:

```powershell
python PPO/evaluate.py --run-name seed42 --episodes 5 --all-checkpoints
```

Evaluate and replay one specific checkpoint (only one model can be rendered at a time):

```powershell
python PPO/evaluate.py --run-name seed42 --model PPO/runs/seed42/checkpoints/ppo_10000_steps.zip --episodes 1 --render
```

Use `A2C/evaluate.py` or `SAC/evaluate.py` for the other algorithms, with their matching run folders. Evaluation uses the same episode seeds across models to make comparisons more consistent.

## Fair comparison

Train PPO, A2C, and SAC using the same environment, timestep budget, and seed before comparing them. The older experiments used a different environment version, so their scores should be treated as historical baselines, not direct apples-to-apples comparisons.
