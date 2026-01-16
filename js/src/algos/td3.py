from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from models import ActorMLP, CriticMLP


@dataclass
class TD3Config:
    obs_dim: int
    action_dim: int
    action_limit: float
    actor_hidden: list[int]
    critic_hidden: list[int]
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    policy_noise: float
    noise_clip: float
    policy_delay: int
    device: torch.device


class TD3Agent:
    def __init__(self, cfg: TD3Config) -> None:
        self.cfg = cfg
        self.device = cfg.device
        self.update_steps = 0

        self.actor = ActorMLP(
            cfg.obs_dim, cfg.action_dim, cfg.action_limit, cfg.actor_hidden
        ).to(self.device)
        self.actor_target = ActorMLP(
            cfg.obs_dim, cfg.action_dim, cfg.action_limit, cfg.actor_hidden
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticMLP(
            cfg.obs_dim, cfg.action_dim, cfg.critic_hidden
        ).to(self.device)
        self.critic2 = CriticMLP(
            cfg.obs_dim, cfg.action_dim, cfg.critic_hidden
        ).to(self.device)
        self.critic1_target = CriticMLP(
            cfg.obs_dim, cfg.action_dim, cfg.critic_hidden
        ).to(self.device)
        self.critic2_target = CriticMLP(
            cfg.obs_dim, cfg.action_dim, cfg.critic_hidden
        ).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=cfg.critic_lr,
        )

    def act(self, obs: np.ndarray, noise_sigma: float = 0.0, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if not deterministic and noise_sigma > 0:
            action = action + noise_sigma * np.random.randn(self.cfg.action_dim)
        action = np.clip(action, -self.cfg.action_limit, self.cfg.action_limit)
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.update_steps += 1

        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"].unsqueeze(-1)
        next_obs = batch["next_obs"]
        dones = batch["dones"].unsqueeze(-1)

        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            noise = torch.randn_like(next_actions) * self.cfg.policy_noise
            noise = noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            next_actions = (next_actions + noise).clamp(
                -self.cfg.action_limit, self.cfg.action_limit
            )

            target_q1 = self.critic1_target(next_obs, next_actions)
            target_q2 = self.critic2_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2).unsqueeze(-1)
            target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

        current_q1 = self.critic1(obs, actions).unsqueeze(-1)
        current_q2 = self.critic2(obs, actions).unsqueeze(-1)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.update_steps % self.cfg.policy_delay == 0:
            actor_actions = self.actor(obs)
            actor_loss = -self.critic1(obs, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)

        return {
            "actor_loss": float(actor_loss.detach().cpu().item()),
            "critic_loss": float(critic_loss.detach().cpu().item()),
        }

    def _soft_update(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        tau = self.cfg.tau
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic1_target.load_state_dict(state["critic1_target"])
        self.critic2_target.load_state_dict(state["critic2_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
