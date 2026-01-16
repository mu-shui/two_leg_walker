import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute convergence metrics from eval.json logs.")
    parser.add_argument("--logdir", default="logs", help="Root logs directory")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Optional run directory names to include (default: all subdirs with eval.json)",
    )
    parser.add_argument("--len-min", type=float, default=300.0)
    parser.add_argument("--speed-min", type=float, default=0.0)
    parser.add_argument("--reward-min", type=float, default=500.0)
    parser.add_argument("--consec", type=int, default=2)
    parser.add_argument("--out", default="", help="Optional output markdown file")
    return parser.parse_args()


def load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_convergence_step(
    eval_data: List[Dict[str, Any]],
    len_min: float,
    speed_min: float,
    reward_min: float,
    consec: int,
) -> Optional[int]:
    streak = 0
    for idx, item in enumerate(eval_data):
        if (
            item.get("eval_len_mean", -1) >= len_min
            and item.get("eval_speed_mean", -1) > speed_min
            and item.get("eval_reward_mean", -1) >= reward_min
        ):
            streak += 1
        else:
            streak = 0
        if streak >= consec:
            first_idx = idx - consec + 1
            return int(eval_data[first_idx].get("step", 0))
    return None


def render_table(rows: List[Dict[str, Any]]) -> str:
    header = (
        "| run_id | eval_random_friction | converge_step | final_len_mean | final_speed_mean "
        "| final_reward_mean | final_reward_std | reward_var |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header]
    for row in rows:
        line = (
            f"| {row['run_id']} | {row['eval_random_friction']} | {row['converge_step']} "
            f"| {row['final_len_mean']} | {row['final_speed_mean']} | {row['final_reward_mean']} "
            f"| {row['final_reward_std']} | {row['reward_var']} |"
        )
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    log_root = Path(args.logdir)
    if args.runs:
        run_dirs = [log_root / name for name in args.runs]
    else:
        run_dirs = [p for p in log_root.iterdir() if p.is_dir()]

    rows = []
    for run_dir in sorted(run_dirs):
        eval_path = run_dir / "eval.json"
        if not eval_path.exists():
            continue
        data = load_json(eval_path)
        if not isinstance(data, list) or not data:
            continue
        data = sorted(data, key=lambda x: x.get("step", 0))
        final = data[-1]
        cfg = load_json(run_dir / "config.json") or {}
        std = float(final.get("eval_reward_std", 0.0))
        rows.append(
            {
                "run_id": run_dir.name,
                "eval_random_friction": cfg.get("eval_random_friction"),
                "converge_step": find_convergence_step(
                    data, args.len_min, args.speed_min, args.reward_min, args.consec
                ),
                "final_len_mean": final.get("eval_len_mean"),
                "final_speed_mean": final.get("eval_speed_mean"),
                "final_reward_mean": final.get("eval_reward_mean"),
                "final_reward_std": final.get("eval_reward_std"),
                "reward_var": round(std * std, 2),
            }
        )

    table = render_table(rows)
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
