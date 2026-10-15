import argparse
from pathlib import Path

import gymnasium as gym
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bottle_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

parser = argparse.ArgumentParser()
parser.add_argument("--timesteps", type=int, default=500_000)
parser.add_argument("--checkpoint-every", type=int, default=10_000)
parser.add_argument("--run-name", default="seed42")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

if args.timesteps <= 0 or args.checkpoint_every <= 0:
    parser.error("Timesteps and checkpoint frequency must be positive.")

# Save outputs inside PPO/runs/<run-name>/
RUN_DIR = Path(__file__).resolve().parent / "runs" / args.run_name
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
TENSORBOARD_DIR = RUN_DIR / "tensorboard"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

env = Monitor(gym.make("BottleDynamics-v0"))

checkpoint_callback = CheckpointCallback(
    save_freq=args.checkpoint_every,
    save_path=str(CHECKPOINT_DIR),
    name_prefix="ppo",
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=str(TENSORBOARD_DIR),
    seed=args.seed,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
)

model.learn(
    total_timesteps=args.timesteps,
    tb_log_name=f"PPO_BottleDynamics_{args.run_name}",
    callback=checkpoint_callback,
)

model.save(str(RUN_DIR / "final_model"))
env.close()

print("\nTraining complete.")
print(f"Final model: {RUN_DIR / 'final_model.zip'}")
print(f"Checkpoints: {CHECKPOINT_DIR}")
print(f"TensorBoard logs: {TENSORBOARD_DIR}")
