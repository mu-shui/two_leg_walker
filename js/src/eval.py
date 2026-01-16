import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import gymnasium as gym

from envs import make_walker2d
from models import ActorMLP
from algos.ppo import GaussianActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DDPG policy.")
    parser.add_argument("--model", required=True, help="Checkpoint .pt path or run directory")
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--random-friction", action="store_true")
    parser.add_argument("--friction-low", type=float, default=0.7)
    parser.add_argument("--friction-high", type=float, default=1.3)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_model_path(path: Path) -> Path:
    if path.is_dir():
        final = path / "final.pt"
        if final.exists():
            return final
        checkpoints = sorted(path.glob("checkpoint_*.pt"), key=lambda p: p.stat().st_mtime)
        if checkpoints:
            return checkpoints[-1]
        raise SystemExit("No checkpoint found in directory.")
    return path


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_video_dir(video_dir: str, model_path: Path) -> Path:
    if video_dir:
        path = Path(video_dir)
        if path.is_absolute():
            return path
        base = model_path if model_path.is_dir() else model_path.parent
        return base / path
    base = model_path if model_path.is_dir() else model_path.parent
    return base / "eval_videos"


def run_episodes(
    policy,
    env: gym.Env,
    episodes: int,
    device: torch.device,
    seed: int,
) -> Tuple[List[float], List[int], List[float]]:
    rewards: List[float] = []
    lengths: List[int] = []
    speeds: List[float] = []
    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_speed_sum = 0.0
        ep_speed_count = 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = policy(obs_t).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)
            x_vel = info.get("x_velocity") if isinstance(info, dict) else None
            if x_vel is not None:
                ep_speed_sum += float(x_vel)
                ep_speed_count += 1
            ep_reward += float(reward)
            ep_len += 1
            done = terminated or truncated
        rewards.append(ep_reward)
        lengths.append(ep_len)
        if ep_speed_count > 0:
            speeds.append(ep_speed_sum / ep_speed_count)
        if ep < episodes - 1:
            obs, _ = env.reset(seed=seed + ep + 1)
    return rewards, lengths, speeds


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    args = parse_args()
    device = resolve_device(args.device)

    model_path = resolve_model_path(Path(args.model))
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})

    eval_env = make_walker2d(
        args.env_id,
        seed=args.seed,
        render_mode=None,
        max_episode_steps=args.max_episode_steps or None,
        random_friction=args.random_friction,
        friction_low=args.friction_low,
        friction_high=args.friction_high,
    )

    obs_dim = eval_env.observation_space.shape[0]
    action_dim = eval_env.action_space.shape[0]
    action_limit = float(np.max(np.abs(eval_env.action_space.high)))
    actor_hidden = config.get("actor_hidden", [256, 128])

    algo = config.get("algo", "ddpg")
    if algo == "ppo":
        actor = GaussianActor(obs_dim, action_dim, action_limit, actor_hidden).to(device)
        actor.load_state_dict(checkpoint["agent"]["actor"])
        actor.eval()

        def policy(obs_t: torch.Tensor) -> torch.Tensor:
            return actor.deterministic(obs_t)
    else:
        actor = ActorMLP(obs_dim, action_dim, action_limit, actor_hidden).to(device)
        actor.load_state_dict(checkpoint["agent"]["actor"])
        actor.eval()

        def policy(obs_t: torch.Tensor) -> torch.Tensor:
            return actor(obs_t)

    rewards, lengths, speeds = run_episodes(
        policy, eval_env, args.episodes, device, args.seed
    )

    eval_env.close()
    mean = float(np.mean(rewards))
    std = float(np.std(rewards))
    mean_len = float(np.mean(lengths))
    if speeds:
        mean_speed = float(np.mean(speeds))
        std_speed = float(np.std(speeds))
    else:
        mean_speed = float("nan")
        std_speed = float("nan")
    print(
        "[eval] episodes={episodes}, mean_reward={mean:.2f}, std={std:.2f}, "
        "mean_len={mean_len:.1f}, mean_speed={mean_speed:.3f}, speed_std={std_speed:.3f}".format(
            episodes=args.episodes,
            mean=mean,
            std=std,
            mean_len=mean_len,
            mean_speed=mean_speed,
            std_speed=std_speed,
        )
    )

    output: Dict[str, Any] = {
        "episodes": args.episodes,
        "mean_reward": mean,
        "std_reward": std,
        "mean_len": mean_len,
        "mean_speed": mean_speed,
        "std_speed": std_speed,
        "model": str(model_path),
        "env_id": args.env_id,
        "random_friction": args.random_friction,
        "max_episode_steps": args.max_episode_steps or None,
    }

    if args.record_video:
        video_dir = resolve_video_dir(args.video_dir, model_path)
        video_dir.mkdir(parents=True, exist_ok=True)
        video_env = make_walker2d(
            args.env_id,
            seed=args.seed,
            render_mode="rgb_array",
            max_episode_steps=args.max_episode_steps or None,
            random_friction=args.random_friction,
            friction_low=args.friction_low,
            friction_high=args.friction_high,
        )
        video_env = gym.wrappers.RecordVideo(
            video_env,
            video_folder=str(video_dir),
            episode_trigger=lambda ep: ep < args.video_episodes,
            name_prefix="eval",
        )
        run_episodes(policy, video_env, args.video_episodes, device, args.seed)
        video_env.close()
        output["videos"] = sorted(p.name for p in video_dir.glob("*.mp4"))

    if args.render:
        render_env = make_walker2d(
            args.env_id,
            seed=args.seed,
            render_mode="human",
            max_episode_steps=args.max_episode_steps or None,
            random_friction=args.random_friction,
            friction_low=args.friction_low,
            friction_high=args.friction_high,
        )
        run_episodes(policy, render_env, args.episodes, device, args.seed)
        render_env.close()

    out_dir = model_path if model_path.is_dir() else model_path.parent
    (out_dir / "eval_metrics.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
