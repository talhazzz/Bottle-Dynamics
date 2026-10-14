from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stable_baselines3 import PPO
from evaluation_utils import evaluate

if __name__ == "__main__":
    evaluate(PPO, "PPO", Path(__file__).resolve().parent)
