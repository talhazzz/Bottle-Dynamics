
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from bottle_env import BottleSwingEnv


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints" / "ppo_platform_v2"
LOG_DIR = PROJECT_DIR / "logs" / "ppo_platform_v2"
TENSORBOARD_DIR = LOG_DIR / "tensorboard"

# First run: verify the training pipeline with a short experiment.
# This is a pilot, not the final training budget.
TOTAL_TIMESTEPS = 100_000

# Save a model snapshot every 5,000 environment steps.
CHECKPOINT_FREQ = 5_000

SEED = 42


def main():
    # Create output folders if they don't already exist.
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)

    # Create the custom Gymnasium environment.
    env = BottleSwingEnv()

    # Stable-Baselines3 callback for periodic model snapshots.
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(CHECKPOINT_DIR),
        name_prefix="bottle_ppo",
        verbose=1,
    )

    # PPO policy and learning configuration.
    model = PPO(
        policy="MlpPolicy",
        env=env,

        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,

        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,

        seed=SEED,
        device="cpu",
        verbose=1,

        tensorboard_log=str(TENSORBOARD_DIR),
    )

    try:
        print("\n" + "=" * 55)
        print("Bottle-Dynamics | PPO Baseline Training")
        print("=" * 55)
        print(f"Training steps:      {TOTAL_TIMESTEPS:,}")
        print(f"Checkpoint interval: {CHECKPOINT_FREQ:,}")
        print(f"Random seed:         {SEED}")
        print(f"Checkpoint folder:   {CHECKPOINT_DIR}")
        print("=" * 55 + "\n")

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            tb_log_name="bottle_ppo_baseline",
        )

        # Save the model at the end as well.
        # We will compare it with the periodic checkpoints later.
        final_model_path = CHECKPOINT_DIR / "bottle_ppo_final"
        model.save(str(final_model_path))

        print("\nTraining finished.")
        print(f"Final model saved to: {final_model_path}.zip")
        print(f"Periodic checkpoints: {CHECKPOINT_DIR}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
