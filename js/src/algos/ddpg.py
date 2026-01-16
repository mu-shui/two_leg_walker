from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import ActorMLP, CriticLSTM, CriticMLP


@dataclass
class DDPGConfig:
    obs_dim: int
    action_dim: int
    action_limit: float
    actor_hidden: list[int]
    critic_hidden: list[int]
    critic_arch: str
    lstm_hidden: int
    lstm_layers: int
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    device: torch.device


class DDPGAgent:
    def __init__(self, cfg: DDPGConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.actor = ActorMLP(
            cfg.obs_dim, cfg.action_dim, cfg.action_limit, cfg.actor_hidden
        ).to(self.device)
        self.actor_target = ActorMLP(
            cfg.obs_dim, cfg.action_dim, cfg.action_limit, cfg.actor_hidden
        ).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        if cfg.critic_arch == "lstm":
            self.critic = CriticLSTM(
                cfg.obs_dim,
                cfg.action_dim,
                cfg.lstm_hidden,
                cfg.lstm_layers,
                cfg.critic_hidden,
            ).to(self.device)
            self.critic_target = CriticLSTM(
                cfg.obs_dim,
                cfg.action_dim,
                cfg.lstm_hidden,
                cfg.lstm_layers,
                cfg.critic_hidden,
            ).to(self.device)
        else:
            self.critic = CriticMLP(
                cfg.obs_dim, cfg.action_dim, cfg.critic_hidden
            ).to(self.device)
            self.critic_target = CriticMLP(
                cfg.obs_dim, cfg.action_dim, cfg.critic_hidden
            ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def act(self, obs: np.ndarray, noise_sigma: float = 0.0, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if not deterministic and noise_sigma > 0:
            action = action + noise_sigma * np.random.randn(self.cfg.action_dim)
        action = np.clip(action, -self.cfg.action_limit, self.cfg.action_limit)
        return action

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        if self.cfg.critic_arch == "lstm":
            obs_seq = batch["obs_seq"]
            actions = batch["actions"]
            rewards = batch["rewards"].unsqueeze(-1)
            next_obs_seq = batch["next_obs_seq"]
            dones = batch["dones"].unsqueeze(-1)

            last_obs = obs_seq[:, -1, :]
            last_next_obs = next_obs_seq[:, -1, :]
            with torch.no_grad():
                next_actions = self.actor_target(last_next_obs)
                target_q = self.critic_target(next_obs_seq, next_actions).unsqueeze(-1)
                target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

            current_q = self.critic(obs_seq, actions).unsqueeze(-1)
            critic_loss = F.mse_loss(current_q, target)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_actions = self.actor(last_obs)
            actor_loss = -self.critic(obs_seq, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
        else:
            obs = batch["obs"]
            actions = batch["actions"]
            rewards = batch["rewards"].unsqueeze(-1)
            next_obs = batch["next_obs"]
            dones = batch["dones"].unsqueeze(-1)

            with torch.no_grad():
                next_actions = self.actor_target(next_obs)
                target_q = self.critic_target(next_obs, next_actions).unsqueeze(-1)
                target = rewards + self.cfg.gamma * (1.0 - dones) * target_q

            current_q = self.critic(obs, actions).unsqueeze(-1)
            critic_loss = F.mse_loss(current_q, target)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            actor_actions = self.actor(obs)
            actor_loss = -self.critic(obs, actor_actions).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

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
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_target.load_state_dict(state["actor_target"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
