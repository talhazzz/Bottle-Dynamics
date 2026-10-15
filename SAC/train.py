import argparse
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

# Register the original Bottle-Dynamics environment.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bottle_env


parser = argparse.ArgumentParser(description="Train SAC on Bottle-Dynamics.")
parser.add_argument("--timesteps", type=int, default=500_000)
parser.add_argument("--checkpoint-every", type=int, default=10_000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--run-name",
    help="Output folder name; defaults to seed<SEED>.",
)
args = parser.parse_args()

if args.timesteps < 1 or args.checkpoint_every < 1:
    parser.error("Timesteps and checkpoint frequency must be positive.")

run_name = args.run_name or f"seed{args.seed}"
run_dir = Path(__file__).resolve().parent / "runs" / run_name

# Never overwrite an existing experiment.
if run_dir.exists() and any(run_dir.iterdir()):
    parser.error(
        f"{run_dir} already contains files. Choose a new --run-name."
    )

checkpoint_dir = run_dir / "checkpoints"
tensorboard_dir = run_dir / "tensorboard"
best_model_dir = run_dir / "best_model"
evaluation_dir = run_dir / "evaluation"

for folder in (
    checkpoint_dir,
    tensorboard_dir,
    best_model_dir,
    evaluation_dir,
):
    folder.mkdir(parents=True)

# Keep training and evaluation environments separate.
env = Monitor(
    gym.make("BottleDynamics-v0"),
    info_keywords=("is_success",),
)
eval_env = Monitor(
    gym.make("BottleDynamics-v0"),
    info_keywords=("is_success",),
)

try:
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(tensorboard_dir),
        seed=args.seed,
        device="cpu",
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=256,
    )

    # Save periodic model checkpoints.
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_every,
        save_path=str(checkpoint_dir),
        name_prefix="sac",
        verbose=1,
    )

    # Evaluate the current policy and save the best one.
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(evaluation_dir),
        eval_freq=args.checkpoint_every,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print("Environment: BottleDynamics-v0")
    print(f"Training steps: {args.timesteps:,}")
    print(f"Checkpoint interval: {args.checkpoint_every:,}")
    print(f"Run directory: {run_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        tb_log_name=run_name,
        callback=CallbackList([
            checkpoint_callback,
            eval_callback,
        ]),
    )

    model.save(str(run_dir / "final_model"))

    print("\nSAC training complete.")
    print(f"Final model: {run_dir / 'final_model.zip'}")
    print(f"Best model: {best_model_dir / 'best_model.zip'}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")

finally:
    env.close()
    eval_env.close()
