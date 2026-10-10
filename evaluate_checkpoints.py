import argparse
import csv
import re
from collections import Counter
from pathlib import Path
from statistics import mean

import pygame
from stable_baselines3 import PPO

from bottle_env import BottleSwingEnv


PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "ppo_platform_v2"
RESULTS_DIR = PROJECT_DIR / "experiments"

DEFAULT_EPISODES = 1
BASE_SEED = 1000
MAX_STEPS_PER_EPISODE = 300


class PlaybackStopped(Exception):
    """Raised when the user closes the visualization window."""


def find_checkpoints():
    """Find periodic checkpoints and the final PPO model."""

    files = list(
        CHECKPOINT_DIR.glob("bottle_ppo_*_steps.zip")
    )

    final_model = CHECKPOINT_DIR / "bottle_ppo_final.zip"

    if final_model.exists():
        files.append(final_model)

    def sort_key(path):
        match = re.search(r"_(\d+)_steps\.zip$", path.name)

        if match:
            return (0, int(match.group(1)))

        return (1, 0)

    return sorted(files, key=sort_key)


def evaluate_episode(model, env, episode_seed, visualize):
    """Evaluate one checkpoint on a single episode."""

    observation, info = env.reset(seed=episode_seed)

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    if visualize:
        env.render()

        if env.window_closed:
            raise PlaybackStopped

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += float(reward)
        steps += 1

        if visualize:
            # Render the actual state after each agent action.
            env.render()

            if env.window_closed:
                raise PlaybackStopped

        if steps >= MAX_STEPS_PER_EPISODE:
            info = dict(info)
            info["outcome"] = "evaluation_step_limit"
            info["success"] = False
            break

    result = {
        "seed": episode_seed,
        "reward": total_reward,
        "success": bool(info["success"]),
        "outcome": info["outcome"],
        "steps": steps,

        "first_contact_angle_error_deg": (
            info["first_contact_angle_error_deg"]
        ),
        "first_contact_angular_velocity_rad_s": (
            info["first_contact_angular_velocity_rad_s"]
        ),
        "first_contact_speed_m_s": (
            info["first_contact_speed_m_s"]
        ),

        "flight_rotation_deg": info["flight_rotation_deg"],
        "net_rotation_deg": info["net_rotation_deg"],
        "final_angle_error_deg": (
            info["landing_angle_error_deg"]
        ),
        "final_angular_velocity_rad_s": (
            float(env.body.angular_velocity)
        ),
        "final_linear_speed_m_s": (
            float(env.body.velocity.length)
        ),
    }

    if visualize:
        # Let us see the final landing before the next episode.
        pygame.time.delay(800)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PPO checkpoints for Bottle-Dynamics."
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Show each evaluation episode in a Pygame window.",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help="Number of evaluation episodes per checkpoint.",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    checkpoints = find_checkpoints()

    if not checkpoints:
        raise FileNotFoundError(
            f"No PPO checkpoints found in {CHECKPOINT_DIR}"
        )

    print("=" * 65)
    print("Bottle-Dynamics | PPO Checkpoint Evaluation")
    print("=" * 65)
    print(f"Models found: {len(checkpoints)}")
    print(f"Episodes per model: {args.episodes}")
    print(f"Pygame visualization: {args.render}")
    print()

    episode_results = []
    summary_results = []

    # Use the same environment instance for all episodes.
    # This keeps one Pygame window open during visual evaluation.
    env = BottleSwingEnv(
        render_mode="human" if args.render else None
    )

    try:
        for checkpoint_path in checkpoints:
            print(f"\nEvaluating: {checkpoint_path.name}")

            model = PPO.load(
                str(checkpoint_path),
                device="cpu",
            )

            model_results = []

            for episode_index in range(args.episodes):
                seed = BASE_SEED + episode_index

                result = evaluate_episode(
                    model=model,
                    env=env,
                    episode_seed=seed,
                    visualize=args.render,
                )

                result["checkpoint"] = checkpoint_path.name

                episode_results.append(result)
                model_results.append(result)

                print(
                    f"  Episode {episode_index + 1}: "
                    f"outcome={result['outcome']}, "
                    f"reward={result['reward']:.3f}, "
                    f"rotation={result['flight_rotation_deg']:.1f} deg"
                )

            successes = sum(
                result["success"]
                for result in model_results
            )

            contact_results = [
                result for result in model_results
                if result["first_contact_angle_error_deg"] is not None
            ]

            outcomes = Counter(
                result["outcome"]
                for result in model_results
            )

            summary = {
                "checkpoint": checkpoint_path.name,
                "episodes": len(model_results),
                "successes": successes,
                "success_rate_pct": (
                    100.0 * successes / len(model_results)
                ),
                "mean_reward": mean(
                    result["reward"]
                    for result in model_results
                ),
                "mean_flight_rotation_deg": mean(
                    result["flight_rotation_deg"]
                    for result in model_results
                ),
                "mean_first_contact_angle_error_deg": (
                    mean(
                        result["first_contact_angle_error_deg"]
                        for result in contact_results
                    )
                    if contact_results else None
                ),
                "mean_first_contact_angular_velocity_rad_s": (
                    mean(
                        result["first_contact_angular_velocity_rad_s"]
                        for result in contact_results
                    )
                    if contact_results else None
                ),
                "outcomes": "; ".join(
                    f"{name}: {count}"
                    for name, count in outcomes.items()
                ),
            }

            summary_results.append(summary)

    except PlaybackStopped:
        print(
            "\nPygame window closed. Evaluation stopped by the user. "
            "Incomplete results were not saved."
        )
        return

    finally:
        env.close()

    # Save individual episode results.
    episode_csv = (
    RESULTS_DIR / "checkpoint_evaluation_episodes_ppo_platform_v2.csv"
)

    with episode_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(episode_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(episode_results)

    # Save one summary row per checkpoint.
    summary_csv = (
    RESULTS_DIR / "checkpoint_evaluation_summary_ppo_platform_v2.csv"
)

    with summary_csv.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_results)

    print("\n" + "=" * 65)
    print("CHECKPOINT SUMMARY")
    print("=" * 65)

    for result in summary_results:
        print(
            f"{result['checkpoint']:32s} "
            f"success={result['success_rate_pct']:5.1f}% "
            f"reward={result['mean_reward']:8.3f} "
            f"rotation={result['mean_flight_rotation_deg']:7.1f} deg"
        )
        print(f"  Outcomes: {result['outcomes']}")

    print(f"\nEpisode results: {episode_csv}")
    print(f"Checkpoint summary: {summary_csv}")

    print(
        "\nNote: initial conditions are currently fixed. "
        "These evaluations do not establish generalization."
    )


if __name__ == "__main__":
    main()
