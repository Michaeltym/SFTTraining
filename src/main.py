from src.baseline import run_baseline
from src.config import MODE, MODE_BASE, MODE_TRAINED

if __name__ == "__main__":
    if MODE == MODE_BASE:
        run_baseline()
    elif MODE == MODE_TRAINED:
        pass
