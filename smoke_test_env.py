
import numpy as np

from bottle_env import BottleSwingEnv


def main():
    env = BottleSwingEnv()

    observation, info = env.reset(seed=42)

    print("=== Bottle-Dynamics Diagnostic ===")
    print("Observation shape:", observation.shape)
    print("Action shape:", env.action_space.shape)
    print("Initial phase:", info["phase"])
    print()

    assert env.observation_space.contains(observation), (
        "Initial observation is outside the observation space."
    )

    total_reward = 0.0
    previous_phase = info["phase"]

    # Maximum of approximately 4 seconds of agent decisions.
    max_steps = 120

    for step in range(max_steps):

        if info["phase"] == "swing":
            # Observation index 2 is normalized angular velocity.
            angular_velocity = float(observation[2])

            # Pump the swing by applying torque in the direction
            # of motion. Near a turning point, push away from the
            # neutral position instead.
            if abs(angular_velocity) > 0.015:
                torque_action = float(np.sign(angular_velocity))
            else:
                torque_action = (
                    1.0 if observation[0] >= 0 else -1.0
                )

            # Request release only after the environment says
            # that its release conditions have been met.
            release_action = (
                1.0 if info["release_ready"] else -1.0
            )

        else:
            # There is no agent control during free flight.
            torque_action = 0.0
            release_action = -1.0

        action = np.array(
            [torque_action, release_action],
            dtype=np.float32,
        )

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        total_reward += reward

        assert env.observation_space.contains(observation), (
            f"Invalid observation at step {step}."
        )

        # Report when the bottle transitions from swing to flight.
        if info["phase"] != previous_phase:
            print(
                f"Step {step}: "
                f"{previous_phase} -> {info['phase']}"
            )

            print(
                "  Flight angular velocity:",
                round(float(observation[2]), 3),
            )

        previous_phase = info["phase"]

        if step % 10 == 0:
            print(
                f"Step {step:3d} | "
                f"Phase: {info['phase']:6s} | "
                f"Angle error: "
                f"{info['landing_angle_error_deg']:.1f} deg | "
                f"Rotation: "
                f"{info['net_rotation_deg']:.1f} deg | "
                f"Reward: {reward:.3f}"
            )

        if terminated or truncated:
            print("\n=== Episode Finished ===")
            print("Outcome:", info["outcome"])
            print("Success:", info["success"])
            print(
                "Final angle error:",
                round(info["landing_angle_error_deg"], 2),
                "degrees",
            )
            print(
                "Total rotation:",
                round(info["net_rotation_deg"], 2),
                "degrees",
            )
            print(
                "Flight rotation:",
                round(info["flight_rotation_deg"], 2),
                "degrees",
            )
            print("Total reward:", round(total_reward, 3))
            print("Steps:", step + 1)
            break

    else:
        print(
            "\nDiagnostic reached its step limit without "
            "a terminal outcome."
        )

    env.close()


if __name__ == "__main__":
    main()
