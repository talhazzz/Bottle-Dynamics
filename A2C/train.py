from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import A2C
from training_utils import train

if __name__ == "__main__":
    train(A2C, "A2C", Path(__file__).resolve().parent,
          n_steps=5, learning_rate=7e-4)
