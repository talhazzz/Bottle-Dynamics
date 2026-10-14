import argparse
from pathlib import Path
import sys

import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bottle_env  # Registers BottleDynamics-v0.


def main():
    parser = argparse.ArgumentParser(description="Train A2C on Bottle-Dynamics.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", help="Output folder name; defaults to seed<SEED>.")
    args = parser.parse_args()

    if args.timesteps < 1 or args.checkpoint_every < 1:
        parser.error("--timesteps and --checkpoint-every must be positive.")

    run_name = args.run_name or f"seed{args.seed}"
    run_dir = Path(__file__).resolve().parent / "runs" / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"{run_dir} already contains files. Choose a new --run-name.")

    checkpoints_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    checkpoints_dir.mkdir(parents=True)
    tensorboard_dir.mkdir()

    env = Monitor(gym.make("BottleDynamics-v0"))
    try:
        model = A2C(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            device="cpu",
            n_steps=5,
                        learning_rate=7e-4,
        )
        checkpoint_callback = CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(checkpoints_dir),
            name_prefix="a2c",
            verbose=1,
        )

        print(f"Training A2C for {args.timesteps:,} timesteps.")
        print(f"Checkpoints every {args.checkpoint_every:,} steps.")
        print(f"Output folder: {run_dir}")

        model.learn(
            total_timesteps=args.timesteps,
            tb_log_name=run_name,
            callback=checkpoint_callback,
        )
        model.save(str(run_dir / "final_model"))

        print(f"Final model: {run_dir / 'final_model.zip'}")
        print(f"Checkpoints: {checkpoints_dir}")
        print(f"TensorBoard logs: {tensorboard_dir}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
