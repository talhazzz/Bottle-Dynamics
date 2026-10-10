
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from bottle_env import BottleSwingEnv


# Each strategy will be tested at several torque strengths
# and requested release times.
STRATEGIES = (
    "pump",
    "constant_positive",
    "constant_negative",
)

TORQUE_STRENGTHS = (0.35, 0.65, 1.0)

RELEASE_REQUEST_TIMES = (
    0.30, 0.45, 0.60, 0.75, 0.90,
    1.05, 1.20, 1.35, 1.50, 1.65,
)

MAX_STEPS = 150


def choose_torque(env, strategy):
    """Return a normalized torque command."""

    angle = env.body.angle
    omega = env.body.angular_velocity

    if strategy == "constant_positive":
        return 1.0

    if strategy == "constant_negative":
        return -1.0

    if strategy == "pump":
        # Add torque in the direction of motion.
        # Near a turning point, push toward the bottom.
        if abs(omega) > 0.05:
            return float(np.sign(omega))

        if abs(angle) > 0.05:
            return float(-np.sign(angle))

        # Break the initial symmetry in a fixed direction.
        return 1.0

    raise ValueError(f"Unknown strategy: {strategy}")


def run_episode(strategy, strength, requested_release_time, seed=42):
    """Run one deterministic controller experiment."""

    env = BottleSwingEnv()
    observation, info = env.reset(seed=seed)

    total_reward = 0.0
    release_time = None
    release_angle = None
    release_omega = None
    release_speed = None

    for step in range(MAX_STEPS):

        if env.phase == "swing":
            torque = strength * choose_torque(env, strategy)

            # Request release only after both the requested time
            # and the environment's minimum swing conditions hold.
            should_release = (
                env.swing_time >= requested_release_time
                and env.release_ready
            )

            release_command = 1.0 if should_release else -1.0

            action = np.array(
                [torque, release_command],
                dtype=np.float32,
            )

        else:
            # The bottle cannot be controlled after release.
            action = np.array([0.0, -1.0], dtype=np.float32)

        old_phase = env.phase

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        # Record the state at the swing-to-flight transition.
        if old_phase == "swing" and info["phase"] == "flight":
            release_time = env.swing_time
            release_angle = math.degrees(env.body.angle)
            release_omega = env.body.angular_velocity
            release_speed = env.body.velocity.length

        if terminated or truncated:
            break

    result = {
        "strategy": strategy,
        "torque_strength": strength,
        "requested_release_time": requested_release_time,
        "actual_release_time": release_time,
        "release_angle_deg": release_angle,
        "release_omega_rad_s": release_omega,
        "release_speed_m_s": release_speed, 
        "outcome": info["outcome"],
        "success": info["success"],
        "final_x_m": float(env.body.position.x),
        "final_y_m": float(env.body.position.y),
        "final_vx_m_s": float(env.body.velocity.x),
        "final_vy_m_s": float(env.body.velocity.y),
        "ground_contact": bool(env.ground_contact),
        "landing_angle_error_deg": info["landing_angle_error_deg"],
        "first_contact_angle_error_deg": (
            info["first_contact_angle_error_deg"]
        ),
        "first_contact_angular_velocity_rad_s": (
            info["first_contact_angular_velocity_rad_s"]
        ),
        "first_contact_speed_m_s": (
            info["first_contact_speed_m_s"]
        ),
        "first_contact_position_m": (
            info["first_contact_position_m"]
        ),
        "net_rotation_deg": info["net_rotation_deg"],
        "flight_rotation_deg": info["flight_rotation_deg"],
        "flight_time_s": info["flight_time"],
        "total_reward": total_reward,
        "steps": step + 1,
        
    }

    env.close()
    return result


def main():
    results = []

    total_trials = (
        len(STRATEGIES)
        * len(TORQUE_STRENGTHS)
        * len(RELEASE_REQUEST_TIMES)
    )

    trial = 0

    print("Bottle-Dynamics: trajectory sweep")
    print(f"Total trials: {total_trials}")
    print()

    for strategy in STRATEGIES:
        for strength in TORQUE_STRENGTHS:
            for release_time in RELEASE_REQUEST_TIMES:
                trial += 1

                result = run_episode(
                    strategy,
                    strength,
                    release_time,
                )
                results.append(result)

                print(
                    f"[{trial:03d}/{total_trials}] "
                    f"{strategy:18s} "
                    f"strength={strength:.2f} "
                    f"request={release_time:.2f}s "
                    f"outcome={result['outcome']:32s} "
                    f"reward={result['total_reward']:.3f}"
                )

    # Store the full experiment table.
    output_dir = Path("experiments")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "trajectory_sweep.csv"

    with output_file.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    # Summarize outcomes.
    counts = Counter(result["outcome"] for result in results)
    successes = sum(result["success"] for result in results)

    print("\n=== Summary ===")
    print(f"Trials: {len(results)}")
    print(f"Successful landings: {successes}")
    print(f"Success rate: {100 * successes / len(results):.1f}%")

    for outcome, count in counts.most_common():
        print(f"{outcome}: {count}")

    # Show the highest-reward trials for inspection.
    print("\n=== Top 10 trials by total reward ===")

    ranked = sorted(
        results,
        key=lambda result: result["total_reward"],
        reverse=True,
    )

    for result in ranked[:10]:
        print(
            f"{result['strategy']:18s} "
            f"strength={result['torque_strength']:.2f} "
            f"requested_release={result['requested_release_time']:.2f}s "
            f"outcome={result['outcome']:32s} "
            f"flight_rotation={result['flight_rotation_deg']:.1f} deg "
            f"angle_error={result['landing_angle_error_deg']:.1f} deg "
            f"reward={result['total_reward']:.3f}"
        )

    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()
