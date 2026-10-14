from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import SAC
from training_utils import train

if __name__ == "__main__":
    train(SAC, "SAC", Path(__file__).resolve().parent,
          learning_rate=3e-4, buffer_size=100_000,
          learning_starts=5_000, batch_size=256)
