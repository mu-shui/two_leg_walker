import argparse
import numpy as np

try:
    import gymnasium as gym
except Exception as exc:  # pragma: no cover - environment import check
    raise SystemExit(f"gymnasium import failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal MuJoCo smoke test.")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ignore-done",
        action="store_true",
        help="Continue stepping even if the episode terminates/truncates.",
    )
    parser.add_argument(
        "--render",
        choices=("none", "human", "rgb_array"),
        default="none",
        help="Render mode for the environment.",
    )
    args = parser.parse_args()

    render_mode = None if args.render == "none" else args.render
    env = gym.make("Walker2d-v4", render_mode=render_mode)
    obs, info = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    total_reward = 0.0
    for _ in range(args.steps):
        action = rng.uniform(env.action_space.low, env.action_space.high)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if render_mode == "human":
            env.render()
        if (terminated or truncated) and not args.ignore_done:
            break

    env.close()
    print(
        f"OK: ran {env.spec.id} for {args.steps} steps, total_reward={total_reward:.2f}"
    )


if __name__ == "__main__":
    main()
