import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward curve from metrics.csv.")
    parser.add_argument("--logdir", required=True, help="Run directory or metrics.csv path")
    parser.add_argument("--out", help="Output image path")
    parser.add_argument("--window", type=int, default=20, help="Rolling window for mean")
    return parser.parse_args()


def find_metrics(path: Path) -> Path:
    if path.is_file() and path.name == "metrics.csv":
        return path
    candidates = list(path.rglob("metrics.csv"))
    if not candidates:
        raise SystemExit("metrics.csv not found in logdir")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_episode_rewards(path: Path):
    steps = []
    rewards = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                actor_loss = float(row.get("actor_loss", 0.0))
                critic_loss = float(row.get("critic_loss", 0.0))
            except ValueError:
                continue
            if actor_loss == 0.0 and critic_loss == 0.0:
                steps.append(int(float(row["step"])))
                rewards.append(float(row["ep_reward"]))
    return np.array(steps), np.array(rewards)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values
    window = min(window, len(values))
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_reward_curve(logdir: Path, out: Path | None = None, window: int = 20) -> Path:
    metrics_path = find_metrics(logdir)
    steps, rewards = load_episode_rewards(metrics_path)

    smooth = rolling_mean(rewards, window)
    smooth_steps = steps[len(steps) - len(smooth) :] if len(smooth) else steps

    plt.figure(figsize=(8, 4.5))
    if len(rewards) > 0:
        plt.plot(steps, rewards, alpha=0.35, label="episode reward")
    if len(smooth) > 0:
        plt.plot(smooth_steps, smooth, label=f"rolling mean ({window})")
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.title("Walker2d Training Reward")
    plt.legend()
    plt.tight_layout()

    out_path = out if out is not None else logdir / "reward_curve.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    print(f"[plot] saved {out_path}")
    return out_path


def main() -> None:
    args = parse_args()
    logdir = Path(args.logdir)
    out_path = Path(args.out) if args.out else None
    plot_reward_curve(logdir, out=out_path, window=args.window)


if __name__ == "__main__":
    main()
