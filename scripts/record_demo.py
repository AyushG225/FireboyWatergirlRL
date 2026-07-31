import argparse

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)

from firewater.firewater_env import record_demo


def parse_args():
    parser = argparse.ArgumentParser(description="Record a keyboard demonstration")
    parser.add_argument("--level", type=int, choices=range(5), default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Record the 19-value legacy state instead of enhanced geometry",
    )
    parser.add_argument(
        "--output",
        help="Output .npz path (default: demo_level<level>_run.npz)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be at least 1")

    output = args.output or f"demo_level{args.level}_run.npz"
    record_demo(
        level_id=args.level,
        max_steps=args.max_steps,
        out_path=output,
        observation_mode="legacy" if args.legacy else "enhanced",
    )


if __name__ == "__main__":
    main()
