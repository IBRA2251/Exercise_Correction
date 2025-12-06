import argparse
import os
from config import EXERCISE_CATALOG
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Run Trainer (record baseline waits).")
    parser.add_argument("--exercise", "-e", default="squat",
                        help=f"Exercise name. Available: {', '.join(EXERCISE_CATALOG.keys())}")
    parser.add_argument("--out", "-o", default="trainer_output",
                        help="Base name for saved files (will create <out>_wait_baseline.json)")
    args = parser.parse_args()

    exercise = args.exercise
    if exercise not in EXERCISE_CATALOG:
        print(f"[Error] Unknown exercise '{exercise}'. Available: {', '.join(EXERCISE_CATALOG.keys())}")
        return

    cfg = EXERCISE_CATALOG[exercise]
    print(f"[RunTrainer] Exercise: {exercise}  using config: {cfg}")

    trainer = Trainer(cfg)
    out_base = args.out
    # ensure output dir exists if they provide a path
    out_dir = os.path.dirname(out_base)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    trainer.run(out_base)

if __name__ == "__main__":
    main()
