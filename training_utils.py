"""Shared training helpers for the PPO, A2C, and SAC folders."""

import argparse
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from torch.utils.tensorboard import SummaryWriter

import bottle_env  # Registers BottleDynamics-v0.


class EpisodeMetrics(BaseCallback):
    """Write exactly two TensorBoard scalars from completed episodes."""

    def __init__(self, log_dir, window=100):
        super().__init__()
        self.log_dir = log_dir
        self.rewards = deque(maxlen=window)
        self.lengths = deque(maxlen=window)
        self.writer = None

    def _on_training_start(self):
        self.writer = SummaryWriter(str(self.log_dir))

    def _on_step(self):
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode:
                self.rewards.append(episode["r"])
                self.lengths.append(episode["l"])
                self.writer.add_scalar(
                    "rollout/ep_rew_mean", np.mean(self.rewards), self.num_timesteps
                )
                self.writer.add_scalar(
                    "rollout/ep_len_mean", np.mean(self.lengths), self.num_timesteps
                )
        return True

    def _on_training_end(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()


def train(Model, algorithm, algorithm_dir, **model_options):
    """Run one uninterrupted training session with periodic snapshots."""
    parser = argparse.ArgumentParser(description=f"Train {algorithm} on Bottle-Dynamics.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", help="Output name; defaults to seed<SEED>.")
    args = parser.parse_args()

    if args.timesteps < 1 or args.checkpoint_every < 1:
        parser.error("--timesteps and --checkpoint-every must be positive.")

    run_name = args.run_name or f"seed{args.seed}"
    run_dir = Path(algorithm_dir) / "runs" / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f"{run_dir} already contains a run. Pick a new --run-name.")

    checkpoints = run_dir / "checkpoints"
    tensorboard = run_dir / "tensorboard"
    checkpoints.mkdir(parents=True)
    tensorboard.mkdir()

    env = Monitor(gym.make("BottleDynamics-v0"))
    model = Model(
        "MlpPolicy", env, seed=args.seed, verbose=1, device="cpu", **model_options
    )
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=args.checkpoint_every,
            save_path=str(checkpoints),
            name_prefix=algorithm.lower(),
            verbose=1,
        ),
        EpisodeMetrics(tensorboard),
    ])

    print(f"{algorithm}: {args.timesteps:,} timesteps; checkpoint every {args.checkpoint_every:,}.")
    print(f"Training once without resetting the learner. Output: {run_dir}")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(str(run_dir / "final_model"))
        print(f"Final model: {run_dir / 'final_model.zip'}")
        print(f"Periodic checkpoints: {checkpoints}")
        print(f"TensorBoard: {tensorboard}")
    finally:
        env.close()
