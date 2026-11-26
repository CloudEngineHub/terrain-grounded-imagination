"""Train a Terrain Grounded Imagination policy for ANYmal-D (the defaults are the full recipe; see
src/tgi_anymal/training/config.py for the options).

    python scripts/train.py --log_dir logs/<run_name> [options]
"""

from tgi_anymal.training.config import parse_args


def main():
    cfg = parse_args()  # before torch and Isaac Lab are imported
    from tgi_anymal.training.trainer import Trainer

    Trainer(cfg).run()


if __name__ == "__main__":
    main()
