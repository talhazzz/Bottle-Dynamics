"""Train PPO, A2C, or SAC and save checkpoints without restarting training."""

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

import bottle_env  # Registers BottleDynamics-v0.


ALGORITHMS = {"PPO": PPO, "A2C": A2C, "SAC": SAC}
ROOT = Path(__file__).resolve().parent


class EpisodeMetrics(BaseCallback):
    """Write only mean episode reward and length to TensorBoard."""

    def __init__(self, log_dir, window=100):
        super().__init__()
        self.log_dir, self.window = log_dir, window
        self.rewards, self.lengths = deque(maxlen=window), deque(maxlen=window)

    def _on_training_start(self):
        self.writer = SummaryWriter(str(self.log_dir))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.rewards.append(episode["r"])
                self.lengths.append(episode["l"])
                self.writer.add_scalar("rollout/ep_rew_mean", np.mean(self.rewards), self.num_timesteps)
                self.writer.add_scalar("rollout/ep_len_mean", np.mean(self.lengths), self.num_timesteps)
        return True

    def _on_training_end(self):
        self.writer.flush()
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train Bottle-Dynamics.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="PPO")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    if args.timesteps < 1 or args.checkpoint_every < 1:
        parser.error("Timesteps and checkpoint interval must be positive.")

    run_dir = args.run_dir or ROOT / "runs" / f"{args.algorithm.lower()}_seed{args.seed}"
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"Run folder is not empty: {run_dir}. Choose a new --run-dir.")
    checkpoints, tensorboard = run_dir / "checkpoints", run_dir / "tensorboard"
    checkpoints.mkdir(parents=True, exist_ok=True)
    tensorboard.mkdir(parents=True, exist_ok=True)

    env = Monitor(gym.make("BottleDynamics-v0"))
    options = {"policy": "MlpPolicy", "env": env, "seed": args.seed, "verbose": 1, "device": "cpu"}
    if args.algorithm == "PPO":
        options.update(n_steps=1024, batch_size=64, n_epochs=10, learning_rate=3e-4)
    elif args.algorithm == "A2C":
        options.update(n_steps=5, learning_rate=7e-4)
    else:
        options.update(buffer_size=100_000, learning_starts=5_000, batch_size=256)

    model = ALGORITHMS[args.algorithm](**options)
    callback = CallbackList([
        CheckpointCallback(save_freq=args.checkpoint_every, save_path=str(checkpoints),
                           name_prefix=args.algorithm.lower(), verbose=1),
        EpisodeMetrics(tensorboard),
    ])
    print(f"{args.algorithm}: {args.timesteps:,} steps; save every {args.checkpoint_every:,}.")
    print(f"One continuous training run. Outputs: {run_dir}")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(str(run_dir / "final_model"))
        print(f"Final model saved to {run_dir / 'final_model.zip'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
