"""Compare and replay saved checkpoints for one algorithm."""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import gymnasium as gym

import bottle_env  # Registers BottleDynamics-v0.


def step_number(path):
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else 10**18


def evaluate_episode(model, env, seed, render):
    obs, _ = env.reset(seed=seed)
    reward_sum = 0.0
    length = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        length += 1
        if render:
            env.render()
            if env.unwrapped.window_closed:
                return None
        if terminated or truncated:
            return reward_sum, length, info


def evaluate(Model, algorithm, algorithm_dir):
    parser = argparse.ArgumentParser(description=f"Evaluate {algorithm} checkpoints.")
    parser.add_argument("--run-name", default="seed42")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", type=Path, help="Evaluate this one model file.")
    parser.add_argument("--render", action="store_true", help="Show Pygame playback; requires --model.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive.")
    if args.render and not args.model:
        parser.error("--render requires --model so only one checkpoint is replayed.")

    run_dir = Path(algorithm_dir) / "runs" / args.run_name
    if args.model:
        models = [args.model if args.model.is_absolute() else Path.cwd() / args.model]
    else:
        models = sorted((run_dir / "checkpoints").glob(f"{algorithm.lower()}_*_steps.zip"),
                        key=step_number)
        final_model = run_dir / "final_model.zip"
        if final_model.exists():
            models.append(final_model)
    if not models:
        parser.error(f"No models found for run {run_dir}.")

    env = gym.make("BottleDynamics-v0", render_mode="human" if args.render else None)
    rows = []
    try:
        for path in models:
            print(f"Evaluating {path.name} ...")
            model = Model.load(str(path), device="cpu")
            rewards, lengths, successes, contacts, outcomes = [], [], [], [], Counter()
            for episode in range(args.episodes):
                result = evaluate_episode(model, env, seed=1000 + episode, render=args.render)
                if result is None:
                    print("Replay stopped by user.")
                    break
                episode_reward, length, info = result
                rewards.append(episode_reward)
                lengths.append(length)
                successes.append(bool(info.get("is_success", False)))
                contacts.append(info.get("first_contact_position_m") is not None)
                outcomes[info.get("outcome", "unknown")] += 1
            if not rewards:
                break
            row = {
                "algorithm": algorithm,
                "model": str(path),
                "episodes": len(rewards),
                "mean_reward": mean(rewards),
                "mean_episode_length": mean(lengths),
                "success_rate": sum(successes) / len(successes),
                "ground_contact_rate": sum(contacts) / len(contacts),
                "outcomes": "; ".join(f"{key}:{value}" for key, value in sorted(outcomes.items())),
            }
            rows.append(row)
            print(f"  mean reward={row['mean_reward']:.2f} | mean length={row['mean_episode_length']:.1f} "
                  f"| success={row['success_rate']:.0%} | ground contact={row['ground_contact_rate']:.0%}")
    finally:
        env.close()

    if rows:
        output = run_dir / ("selected_model_evaluation.csv" if args.model else "evaluation_results.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Results saved to {output}")
        if not args.model:
            best = max(rows, key=lambda row: (row["success_rate"], row["mean_reward"]))
            print(f"Top candidate (success rate, then reward): {Path(best['model']).name}")


__all__ = ["evaluate"]
