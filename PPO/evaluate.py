
import argparse
import sys
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bottle_env


parser = argparse.ArgumentParser()
parser.add_argument("--run-name", default="ppo_100k_tensorboard")
parser.add_argument("--model", type=Path, help="Path to a checkpoint or best_model.zip")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--render", action="store_true")
args = parser.parse_args()

if args.episodes < 1:
    parser.error("Episodes must be positive.")

RUN_DIR = Path(__file__).resolve().parent / "runs" / args.run_name

# Use a specific model if supplied; otherwise use the final model.
if args.model:
    MODEL_PATH = args.model
    if not MODEL_PATH.is_absolute():
        MODEL_PATH = Path.cwd() / MODEL_PATH
else:
    MODEL_PATH = RUN_DIR / "final_model.zip"

if not MODEL_PATH.is_file():
    parser.error(f"Model not found: {MODEL_PATH}")

env = gym.make(
    "BottleDynamics-v0",
    render_mode="human" if args.render else None,
)
model = PPO.load(str(MODEL_PATH), device="cpu")

# Keep manual evaluation logs separate from training logs.
writer = SummaryWriter(str(RUN_DIR / "tensorboard" / "manual_evaluation"))

total_rewards = []
successful_episodes = 0

try:
    for episode in range(args.episodes):
        observation, info = env.reset(seed=1000 + episode)
        total_reward = 0.0
        step_count = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            if args.render:
                env.render()
                if env.unwrapped.window_closed:
                    print("Playback stopped by user.")
                    break

        if args.render and env.unwrapped.window_closed:
            break

        success = bool(info.get("is_success", False))
        total_rewards.append(total_reward)
        successful_episodes += int(success)
        success_rate = successful_episodes / len(total_rewards)

        print(
            f"Episode {episode + 1}: "
            f"reward={total_reward:.2f}, "
            f"steps={step_count}, "
            f"success={success}"
        )

        writer.add_scalar("manual_eval/episode_reward", total_reward, episode + 1)
        writer.add_scalar("manual_eval/episode_length", step_count, episode + 1)
        writer.add_scalar("manual_eval/success_rate", success_rate, episode + 1)

    if total_rewards:
        average_reward = sum(total_rewards) / len(total_rewards)
        success_rate = successful_episodes / len(total_rewards)

        print("\n--- Evaluation Summary ---")
        print(f"Model: {MODEL_PATH}")
        print(f"Average reward: {average_reward:.2f}")
        print(f"Success rate: {success_rate:.0%}")
        print(f"Successful episodes: {successful_episodes}/{len(total_rewards)}")

        writer.add_scalar("manual_eval/average_reward", average_reward, len(total_rewards))
        writer.add_scalar("manual_eval/final_success_rate", success_rate, len(total_rewards))
finally:
    writer.close()
    env.close()
