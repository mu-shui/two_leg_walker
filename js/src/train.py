import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

from algos import DDPGAgent, DDPGConfig, TD3Agent, TD3Config, PPOAgent, PPOConfig
from envs import make_walker2d
from utils import CSVLogger, ReplayBuffer, SequenceReplayBuffer, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG/TD3/PPO on Walker2d with MLP/LSTM critic.")
    parser.add_argument("--env-id", default="Walker2d-v4")
    parser.add_argument("--algo", choices=("ddpg", "td3", "ppo"), default="ddpg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--start-steps", type=int, default=10_000)
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--noise-sigma", type=float, default=0.1)
    parser.add_argument("--noise-decay", type=float, default=0.0)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--ppo-steps", type=int, default=2048)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--ppo-minibatch", type=int, default=64)
    parser.add_argument("--ppo-actor-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-critic-lr", type=float, default=1e-3)
    parser.add_argument("--ppo-target-kl", type=float, default=0.01)
    parser.add_argument(
        "--ppo-clip-vloss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.97)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--critic-arch", choices=("mlp", "lstm"), default="mlp")
    parser.add_argument("--actor-hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--random-friction", action="store_true")
    parser.add_argument("--friction-low", type=float, default=0.7)
    parser.add_argument("--friction-high", type=float, default=1.3)
    parser.add_argument("--max-episode-steps", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-random-friction", action="store_true")
    parser.add_argument("--plot-after", action="store_true")
    parser.add_argument("--visualize-after", action="store_true")
    parser.add_argument("--render-episodes", type=int, default=1)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-episodes", type=int, default=1)
    parser.add_argument("--video-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--savedir", default="checkpoints")
    return parser.parse_args()


def build_run_id(args: argparse.Namespace) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{args.algo}_{args.critic_arch}_seed{args.seed}_{ts}"


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate(agent, env, episodes: int, noise_sigma: float) -> Dict[str, float]:
    rewards = []
    lengths = []
    speeds = []
    obs, _ = env.reset()
    for _ in range(episodes):
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_speed_sum = 0.0
        ep_speed_count = 0
        while not done:
            action = agent.act(obs, noise_sigma=noise_sigma, deterministic=True)
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
        obs, _ = env.reset()
    metrics = {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "eval_len_mean": float(np.mean(lengths)),
    }
    if speeds:
        metrics["eval_speed_mean"] = float(np.mean(speeds))
        metrics["eval_speed_std"] = float(np.std(speeds))
    return metrics


def render_demo(
    agent,
    env_id: str,
    seed: int,
    episodes: int,
    max_episode_steps: Optional[int],
    random_friction: bool,
    friction_low: float,
    friction_high: float,
) -> Dict[str, float]:
    env = make_walker2d(
        env_id,
        seed=seed,
        render_mode="human",
        max_episode_steps=max_episode_steps,
        random_friction=random_friction,
        friction_low=friction_low,
        friction_high=friction_high,
    )
    rewards = []
    lengths = []
    speeds = []
    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_speed_sum = 0.0
        ep_speed_count = 0
        while not done:
            action = agent.act(obs, noise_sigma=0.0, deterministic=True)
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
    env.close()
    metrics = {
        "render_reward_mean": float(np.mean(rewards)),
        "render_reward_std": float(np.std(rewards)),
        "render_len_mean": float(np.mean(lengths)),
    }
    if speeds:
        metrics["render_speed_mean"] = float(np.mean(speeds))
        metrics["render_speed_std"] = float(np.std(speeds))
    return metrics


def resolve_video_dir(video_dir: str, log_path: Path) -> Path:
    if not video_dir:
        return log_path / "videos"
    path = Path(video_dir)
    if path.is_absolute():
        return path
    return log_path / path


def record_video(
    agent,
    env_id: str,
    seed: int,
    episodes: int,
    max_episode_steps: Optional[int],
    random_friction: bool,
    friction_low: float,
    friction_high: float,
    video_dir: Path,
) -> Dict[str, float]:
    video_dir.mkdir(parents=True, exist_ok=True)
    env = make_walker2d(
        env_id,
        seed=seed,
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        random_friction=random_friction,
        friction_low=friction_low,
        friction_high=friction_high,
    )
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(video_dir),
        episode_trigger=lambda _ep: True,
        name_prefix="demo",
    )
    rewards = []
    lengths = []
    speeds = []
    obs, _ = env.reset(seed=seed)
    for ep in range(episodes):
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_speed_sum = 0.0
        ep_speed_count = 0
        while not done:
            action = agent.act(obs, noise_sigma=0.0, deterministic=True)
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
    env.close()
    metrics = {
        "video_reward_mean": float(np.mean(rewards)),
        "video_reward_std": float(np.std(rewards)),
        "video_len_mean": float(np.mean(lengths)),
    }
    if speeds:
        metrics["video_speed_mean"] = float(np.mean(speeds))
        metrics["video_speed_std"] = float(np.std(speeds))
    return metrics


def compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros(len(rewards), dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        adv[t] = last_gae
    returns = adv + np.array(values, dtype=np.float32)
    return adv, returns


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "glfw")
    args = parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    run_id = build_run_id(args)
    log_path = Path(args.logdir) / run_id
    save_path = Path(args.savedir) / run_id
    log_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)

    config = vars(args)
    config["run_id"] = run_id
    (log_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (save_path / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = make_walker2d(
        args.env_id,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps or None,
        random_friction=args.random_friction,
        friction_low=args.friction_low,
        friction_high=args.friction_high,
    )

    eval_env = make_walker2d(
        args.env_id,
        seed=args.seed + 10_000,
        max_episode_steps=args.max_episode_steps or None,
        random_friction=args.eval_random_friction,
        friction_low=args.friction_low,
        friction_high=args.friction_high,
    )

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_limit = float(np.max(np.abs(env.action_space.high)))

    if args.algo == "ppo":
        if args.critic_arch != "mlp":
            raise SystemExit("PPO only supports --critic-arch mlp in this codebase.")
        cfg = PPOConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_limit=action_limit,
            actor_hidden=args.actor_hidden,
            critic_hidden=args.critic_hidden,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            actor_lr=args.ppo_actor_lr,
            critic_lr=args.ppo_critic_lr,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            target_kl=args.ppo_target_kl,
            clip_vloss=args.ppo_clip_vloss,
            device=device,
        )
        agent = PPOAgent(cfg)
        buffer = None
    elif args.algo == "td3":
        if args.critic_arch != "mlp":
            raise SystemExit("TD3 only supports --critic-arch mlp in this codebase.")
        cfg = TD3Config(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_limit=action_limit,
            actor_hidden=args.actor_hidden,
            critic_hidden=args.critic_hidden,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            device=device,
        )
        agent = TD3Agent(cfg)
        buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            size=args.replay_size,
            device=device,
        )
    else:
        cfg = DDPGConfig(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_limit=action_limit,
            actor_hidden=args.actor_hidden,
            critic_hidden=args.critic_hidden,
            critic_arch=args.critic_arch,
            lstm_hidden=args.lstm_hidden,
            lstm_layers=args.lstm_layers,
            gamma=args.gamma,
            tau=args.tau,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            device=device,
        )
        agent = DDPGAgent(cfg)

        if args.critic_arch == "lstm":
            buffer = SequenceReplayBuffer(
                obs_dim=obs_dim,
                action_dim=action_dim,
                max_transitions=args.replay_size,
                seq_len=args.seq_len,
                device=device,
            )
        else:
            buffer = ReplayBuffer(
                obs_dim=obs_dim,
                action_dim=action_dim,
                size=args.replay_size,
                device=device,
            )

    logger = CSVLogger(
        log_path / "metrics.csv",
        ["step", "episode", "ep_reward", "ep_len", "actor_loss", "critic_loss"],
    )

    obs, _ = env.reset(seed=args.seed)
    episode = 0
    ep_reward = 0.0
    ep_len = 0

    if args.algo == "ppo":
        global_step = 0
        with trange(args.total_steps, desc="training") as pbar:
            while global_step < args.total_steps:
                rollout_obs = []
                rollout_actions = []
                rollout_logp = []
                rollout_values = []
                rollout_rewards = []
                rollout_dones = []

                for _ in range(args.ppo_steps):
                    if global_step >= args.total_steps:
                        break
                    action, logp, value = agent.get_action_and_value(obs)
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    rollout_obs.append(obs)
                    rollout_actions.append(action)
                    rollout_logp.append(logp)
                    rollout_values.append(value)
                    rollout_rewards.append(float(reward))
                    rollout_dones.append(done)

                    obs = next_obs
                    ep_reward += float(reward)
                    ep_len += 1
                    global_step += 1
                    pbar.update(1)

                    if done:
                        logger.log(
                            {
                                "step": global_step,
                                "episode": episode,
                                "ep_reward": ep_reward,
                                "ep_len": ep_len,
                                "actor_loss": 0.0,
                                "critic_loss": 0.0,
                            }
                        )
                        episode += 1
                        obs, _ = env.reset(seed=args.seed + episode)
                        ep_reward = 0.0
                        ep_len = 0

                    if args.eval_every > 0 and global_step % args.eval_every == 0:
                        eval_metrics = evaluate(
                            agent, eval_env, args.eval_episodes, noise_sigma=0.0
                        )
                        eval_path = log_path / "eval.json"
                        if eval_path.exists():
                            data = json.loads(eval_path.read_text(encoding="utf-8"))
                        else:
                            data = []
                        data.append({"step": global_step, **eval_metrics})
                        eval_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

                with torch.no_grad():
                    last_value = agent.critic(
                        torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    ).cpu().numpy()[0]
                adv, returns = compute_gae(
                    rollout_rewards,
                    rollout_values,
                    rollout_dones,
                    float(last_value),
                    args.gamma,
                    args.gae_lambda,
                )

                obs_t = torch.as_tensor(np.array(rollout_obs), dtype=torch.float32, device=device)
                actions_t = torch.as_tensor(
                    np.array(rollout_actions), dtype=torch.float32, device=device
                )
                logp_t = torch.as_tensor(np.array(rollout_logp), dtype=torch.float32, device=device)
                returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)
                adv_t = torch.as_tensor(adv, dtype=torch.float32, device=device)
                values_t = torch.as_tensor(
                    np.array(rollout_values), dtype=torch.float32, device=device
                )

                losses = agent.update(
                    obs_t,
                    actions_t,
                    logp_t,
                    returns_t,
                    adv_t,
                    values_t,
                    args.ppo_epochs,
                    args.ppo_minibatch,
                )
                logger.log(
                    {
                        "step": global_step,
                        "episode": episode,
                        "ep_reward": ep_reward,
                        "ep_len": ep_len,
                        "actor_loss": losses["actor_loss"],
                        "critic_loss": losses["critic_loss"],
                    }
                )

                if global_step % 100_000 == 0:
                    checkpoint = {
                        "step": global_step,
                        "config": config,
                        "agent": agent.state_dict(),
                    }
                    torch.save(checkpoint, save_path / f"checkpoint_{global_step}.pt")

        checkpoint = {"step": global_step, "config": config, "agent": agent.state_dict()}
        torch.save(checkpoint, save_path / "final.pt")
    else:
        noise_sigma = args.noise_sigma
        for step in trange(args.total_steps, desc="training"):
            if step < args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs, noise_sigma=noise_sigma)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            buffer.add(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += float(reward)
            ep_len += 1

            if done:
                logger.log(
                    {
                        "step": step + 1,
                        "episode": episode,
                        "ep_reward": ep_reward,
                        "ep_len": ep_len,
                        "actor_loss": 0.0,
                        "critic_loss": 0.0,
                    }
                )
                episode += 1
                obs, _ = env.reset(seed=args.seed + episode)
                ep_reward = 0.0
                ep_len = 0

            if step >= args.update_after and buffer.ready(args.batch_size):
                batch = buffer.sample(args.batch_size)
                losses = agent.update(batch)
                logger.log(
                    {
                        "step": step + 1,
                        "episode": episode,
                        "ep_reward": ep_reward,
                        "ep_len": ep_len,
                        "actor_loss": losses["actor_loss"],
                        "critic_loss": losses["critic_loss"],
                    }
                )

            if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
                eval_metrics = evaluate(agent, eval_env, args.eval_episodes, noise_sigma=0.0)
                eval_path = log_path / "eval.json"
                if eval_path.exists():
                    data = json.loads(eval_path.read_text(encoding="utf-8"))
                else:
                    data = []
                data.append({"step": step + 1, **eval_metrics})
                eval_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

            if args.noise_decay > 0:
                noise_sigma = max(0.0, noise_sigma - args.noise_decay)

            if (step + 1) % 100_000 == 0:
                checkpoint = {
                    "step": step + 1,
                    "config": config,
                    "agent": agent.state_dict(),
                }
                torch.save(checkpoint, save_path / f"checkpoint_{step + 1}.pt")

        checkpoint = {"step": args.total_steps, "config": config, "agent": agent.state_dict()}
        torch.save(checkpoint, save_path / "final.pt")

    logger.close()
    env.close()
    eval_env.close()
    if args.plot_after or args.visualize_after or args.record_video:
        try:
            from plot import plot_reward_curve

            plot_reward_curve(log_path)
        except Exception as exc:
            print(f"[train] plot failed: {exc}")
    if args.visualize_after:
        try:
            render_metrics = render_demo(
                agent,
                args.env_id,
                args.seed + 12345,
                args.render_episodes,
                args.max_episode_steps or None,
                args.random_friction,
                args.friction_low,
                args.friction_high,
            )
            (log_path / "render_eval.json").write_text(
                json.dumps(render_metrics, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[train] render failed: {exc}")
    if args.record_video:
        try:
            video_dir = resolve_video_dir(args.video_dir, log_path)
            video_metrics = record_video(
                agent,
                args.env_id,
                args.seed + 54321,
                args.video_episodes,
                args.max_episode_steps or None,
                args.random_friction,
                args.friction_low,
                args.friction_high,
                video_dir,
            )
            video_files = sorted(p.name for p in video_dir.glob("*.mp4"))
            payload = {"videos": video_files, **video_metrics}
            (log_path / "video_eval.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[train] video record failed: {exc}")
    print(f"[train] done. logs={log_path}, checkpoints={save_path}")


if __name__ == "__main__":
    main()
