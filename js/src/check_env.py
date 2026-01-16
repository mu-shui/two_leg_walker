"""
Quick environment smoke test for the Walker2d course project.
- Verifies Gymnasium + MuJoCo + Torch can import.
- Runs a short seeded rollout with a random policy to ensure the simulator steps without errors.
"""

import argparse
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch


@dataclass
class Config:
    env_id: str = "Walker2d-v4"
    steps: int = 200
    seed: int = 42
    render: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg: Config) -> None:
    render_mode = "human" if cfg.render else None
    env = gym.make(cfg.env_id, render_mode=render_mode)
    obs, info = env.reset(seed=cfg.seed)

    total_reward = 0.0
    episodes = 0
    for t in range(cfg.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            episodes += 1
            obs, info = env.reset(seed=cfg.seed + episodes)

    env.close()
    mean_reward = total_reward / cfg.steps
    print(
        f"[check_env] env={cfg.env_id}, steps={cfg.steps}, "
        f"seed={cfg.seed}, episodes={episodes}, mean_reward={mean_reward:.3f}"
    )


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Walker2d environment smoke test")
    parser.add_argument("--steps", type=int, default=Config.steps, help="Number of steps to run")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed for env and libs")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering if supported")
    args = parser.parse_args()
    return Config(steps=args.steps, seed=args.seed, render=args.render)


if __name__ == "__main__":
    # For headless Linux servers with GPU rendering available, set MUJOCO_GL=egl.
    os.environ.setdefault("MUJOCO_GL", "glfw")

    cfg = parse_args()
    set_seed(cfg.seed)
    run(cfg)
