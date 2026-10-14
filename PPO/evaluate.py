import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import gymnasium as gym
from stable_baselines3 import PPO
from torch.utils.tensorboard import SummaryWriter

# Add the project root so bottle_env.py can be imported when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bottle_env  # Registers BottleDynamics-v0.


def checkpoint_step(path):
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else 10**18


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model.")
    parser.add_argument("--run-name", default="seed42")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model", type=Path, help="Evaluate one model file instead of final_model.zip.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate every saved checkpoint and the final model.")
    parser.add_argument("--render", action="store_true", help="Show the Pygame simulation.")
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be positive.")
    if args.model and args.all_checkpoints:
        parser.error("Use either --model or --all-checkpoints, not both.")
    if args.render and args.all_checkpoints:
        parser.error("--render can only be used with one model.")

    run_dir = Path(__file__).resolve().parent / "runs" / args.run_name
    if args.model:
        model_paths = [args.model if args.model.is_absolute() else Path.cwd() / args.model]
        result_filename = "selected_model_evaluation.csv"
    elif args.all_checkpoints:
        model_paths = sorted(
            (run_dir / "checkpoints").glob("ppo_*_steps.zip"),
            key=checkpoint_step,
        )
        final_model = run_dir / "final_model.zip"
        if final_model.exists():
            model_paths.append(final_model)
        result_filename = "evaluation_results.csv"
    else:
        model_paths = [run_dir / "final_model.zip"]
        result_filename = "evaluation_results.csv"

    model_paths = [path.resolve() for path in model_paths if path.is_file()]
    if not model_paths:
        parser.error(f"No model files found in {run_dir}. Train a model first.")

    env = gym.make("BottleDynamics-v0", render_mode="human" if args.render else None)
    rows = []
    stopped_by_user = False
    writer = SummaryWriter(str(run_dir / "evaluation_tensorboard"))
    try:
        for model_path in model_paths:
            print(f"Evaluating {model_path.name} ...")
            model = PPO.load(str(model_path), device="cpu")
            rewards, lengths, successes, contacts = [], [], [], []
            outcomes = Counter()
            tag = "evaluation" if len(model_paths) == 1 else f"evaluation/{model_path.stem}"

            for episode in range(args.episodes):
                observation, info = env.reset(seed=1000 + episode)
                total_reward = 0.0
                step_count = 0

                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    step_count += 1

                    if args.render:
                        env.render()
                        if env.unwrapped.window_closed:
                            stopped_by_user = True
                            break

                    if terminated or truncated:
                        break

                if stopped_by_user:
                    print("Replay stopped by user.")
                    break

                success = bool(info.get("is_success", False))
                outcome = info.get("outcome", "unknown")
                rewards.append(total_reward)
                lengths.append(step_count)
                successes.append(success)
                contacts.append(info.get("first_contact_position_m") is not None)
                outcomes[outcome] += 1
                success_rate = sum(successes) / len(successes)

                print(
                    f"Episode {episode + 1}: reward={total_reward:.2f}, "
                    f"steps={step_count}, terminated={terminated}, "
                    f"truncated={truncated}, success={success}, outcome={outcome}"
                )
                writer.add_scalar(f"{tag}/episode_reward", total_reward, episode + 1)
                writer.add_scalar(f"{tag}/episode_length", step_count, episode + 1)
                writer.add_scalar(f"{tag}/success_rate", success_rate, episode + 1)

            if rewards:
                row = {
                    "algorithm": "PPO",
                    "model": str(model_path),
                    "episodes": len(rewards),
                    "average_reward": mean(rewards),
                    "average_episode_length": mean(lengths),
                    "success_rate": sum(successes) / len(successes),
                    "ground_contact_rate": sum(contacts) / len(contacts),
                    "outcomes": "; ".join(
                        f"{name}:{count}" for name, count in sorted(outcomes.items())
                    ),
                }
                rows.append(row)
                print("\n*** Evaluation summary ***")
                print(f"Average reward: {row['average_reward']:.2f}")
                print(f"Success rate: {row['success_rate']:.0%}")
                print(f"Successful episodes: {sum(successes)}/{len(successes)}")
                writer.add_scalar(f"{tag}/average_reward", row["average_reward"], len(rewards))
                writer.add_scalar(f"{tag}/final_success_rate", row["success_rate"], len(rewards))
                print(
                    f"Mean length: {row['average_episode_length']:.1f} | "
                    f"Ground contact rate: {row['ground_contact_rate']:.0%}"
                )

            if stopped_by_user:
                break
    finally:
        writer.close()
        env.close()

    if rows:
        run_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / result_filename
        with output_path.open("w", newline="", encoding="utf-8") as file:
            result_writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            result_writer.writeheader()
            result_writer.writerows(rows)
        print(f"Results saved to {output_path}")
        print(f"Evaluation TensorBoard logs: {run_dir / 'evaluation_tensorboard'}")


if __name__ == "__main__":
    main()
