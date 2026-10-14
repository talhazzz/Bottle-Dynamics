import argparse
import csv
import re
import sys
from pathlib import Path
from statistics import mean

import gymnasium as gym
from stable_baselines3 import A2C
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bottle_env  # Registers BottleDynamics-v0.


def checkpoint_step(path):
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else 10**18


def evaluate_model(model_path, env, episodes, writer, tag):
    model = A2C.load(str(model_path), device="cpu")
    rewards, lengths, successes = [], [], []

    for episode in range(episodes):
        observation, _ = env.reset(seed=1000 + episode)
        total_reward = 0.0
        step_count = 0

        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            step_count += 1

            if env.render_mode == "human":
                env.render()
                if env.unwrapped.window_closed:
                    print("Replay stopped by user.")
                    return None

            if terminated or truncated:
                break

        success = bool(info.get("is_success", False))
        rewards.append(total_reward)
        lengths.append(step_count)
        successes.append(success)
        success_rate = sum(successes) / len(successes)

        print(
            f"Episode {episode + 1}: reward={total_reward:.2f}, "
            f"steps={step_count}, terminated={terminated}, "
            f"truncated={truncated}, success={success}"
        )
        writer.add_scalar(f"{tag}/episode_reward", total_reward, episode + 1)
        writer.add_scalar(f"{tag}/episode_length", step_count, episode + 1)
        writer.add_scalar(f"{tag}/success_rate", success_rate, episode + 1)

    row = {
        "algorithm": "A2C",
        "model": str(model_path),
        "episodes": len(rewards),
        "average_reward": mean(rewards),
        "average_episode_length": mean(lengths),
        "success_rate": sum(successes) / len(successes),
        "successful_episodes": f"{sum(successes)}/{len(successes)}",
    }
    print("\n*** Evaluation summary ***")
    print(f"Average reward: {row['average_reward']:.2f}")
    print(f"Average episode length: {row['average_episode_length']:.1f}")
    print(f"Success rate: {row['success_rate']:.0%}")
    print(f"Successful episodes: {row['successful_episodes']}")
    writer.add_scalar(f"{tag}/average_reward", row["average_reward"], len(rewards))
    writer.add_scalar(f"{tag}/final_success_rate", row["success_rate"], len(rewards))
    return row


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained A2C model.")
    parser.add_argument("--run-name", default="seed42")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--model", type=Path, help="Evaluate this model instead of final_model.zip.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Evaluate every checkpoint and the final model.")
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
        result_file = "selected_model_evaluation.csv"
    elif args.all_checkpoints:
        model_paths = sorted(
            (run_dir / "checkpoints").glob("a2c_*_steps.zip"),
            key=checkpoint_step,
        )
        final_model = run_dir / "final_model.zip"
        if final_model.exists():
            model_paths.append(final_model)
        result_file = "evaluation_results.csv"
    else:
        model_paths = [run_dir / "final_model.zip"]
        result_file = "evaluation_results.csv"

    model_paths = [path.resolve() for path in model_paths if path.is_file()]
    if not model_paths:
        parser.error(f"No model found. Check the run folder: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make("BottleDynamics-v0", render_mode="human" if args.render else None)
    rows = []
    writer = SummaryWriter(str(run_dir / "evaluation_tensorboard"))

    try:
        for model_path in model_paths:
            print(f"\nEvaluating {model_path.name} ...")
            tag = "evaluation" if len(model_paths) == 1 else f"evaluation/{model_path.stem}"
            result = evaluate_model(model_path, env, args.episodes, writer, tag)
            if result is None:
                break
            rows.append(result)
    finally:
        writer.close()
        env.close()

    if rows:
        output_path = run_dir / result_file
        with output_path.open("w", newline="", encoding="utf-8") as file:
            csv_writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            csv_writer.writeheader()
            csv_writer.writerows(rows)
        print(f"Results saved to {output_path}")
        print(f"Evaluation TensorBoard logs: {run_dir / 'evaluation_tensorboard'}")


if __name__ == "__main__":
    main()
