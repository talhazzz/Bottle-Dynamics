"""Evaluate saved checkpoints or replay one selected model."""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import gymnasium as gym
from stable_baselines3 import A2C, PPO, SAC

import bottle_env  # Registers BottleDynamics-v0.


ALGORITHMS = {"PPO": PPO, "A2C": A2C, "SAC": SAC}


def step_number(path):
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    return int(match.group(1)) if match else 10**18


def evaluate(model, env, episodes, render):
    rewards, lengths, successes, contacts, outcomes = [], [], [], [], Counter()
    for episode in range(episodes):
        obs, _ = env.reset(seed=1000 + episode)  # Same seeds for every model.
        total, steps = 0.0, 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            steps += 1
            if render:
                env.render()
                if env.unwrapped.window_closed:
                    return None
            if terminated or truncated:
                break
        rewards.append(total)
        lengths.append(steps)
        successes.append(bool(info["is_success"]))
        contacts.append(info["first_contact_position_m"] is not None)
        outcomes[info["outcome"]] += 1
    return {
        "episodes": episodes,
        "mean_reward": mean(rewards),
        "mean_episode_length": mean(lengths),
        "success_rate": sum(successes) / episodes,
        "ground_contact_rate": sum(contacts) / episodes,
        "outcomes": "; ".join(f"{k}:{v}" for k, v in sorted(outcomes.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Bottle-Dynamics checkpoints.")
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--model", type=Path, help="Evaluate one model .zip file.")
    parser.add_argument("--render", action="store_true", help="Replay the selected --model.")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive.")
    if args.render and not args.model:
        parser.error("--render requires --model.")

    if args.model:
        models = [args.model]
    else:
        models = sorted((args.run_dir / "checkpoints").glob(f"{args.algorithm.lower()}_*_steps.zip"),
                        key=step_number)
        final_model = args.run_dir / "final_model.zip"
        if final_model.exists():
            models.append(final_model)
    if not models:
        parser.error("No models found; check --run-dir or --model.")

    env = gym.make("BottleDynamics-v0", render_mode="human" if args.render else None)
    results = []
    try:
        for path in models:
            print(f"Evaluating {path.name} ...")
            model = ALGORITHMS[args.algorithm].load(str(path), device="cpu")
            result = evaluate(model, env, args.episodes, args.render)
            if result is None:
                print("Replay stopped.")
                break
            row = {"algorithm": args.algorithm, "model": path.name, **result}
            results.append(row)
            print(f"  reward={result['mean_reward']:.2f}, length={result['mean_episode_length']:.1f}, "
                  f"success={result['success_rate']:.0%}, contact={result['ground_contact_rate']:.0%}")
    finally:
        env.close()

    if results and not args.model:
        output = args.run_dir / "evaluation_results.csv"
        with output.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=results[0])
            writer.writeheader()
            writer.writerows(results)
        best = max(results, key=lambda row: (row["success_rate"], row["mean_reward"]))
        print(f"Results saved to {output}\nTop candidate: {best['model']}")


if __name__ == "__main__":
    main()
