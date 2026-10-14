from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from training_utils import train

if __name__ == "__main__":
    train(PPO, "PPO", Path(__file__).resolve().parent,
          n_steps=1024, batch_size=64, n_epochs=10, learning_rate=3e-4)
