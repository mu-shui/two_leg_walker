from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.actor import build_mlp


@dataclass
class PPOConfig:
    obs_dim: int
    action_dim: int
    action_limit: float
    actor_hidden: list[int]
    critic_hidden: list[int]
    gamma: float
    gae_lambda: float
    clip_ratio: float
    actor_lr: float
    critic_lr: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    clip_vloss: bool
    target_kl: float
    device: torch.device


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_limit: float, hidden: list[int]) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.action_limit = float(action_limit)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.net(obs)
        log_std = self.log_std.expand_as(mean)
        return mean, log_std

    def _squash_action(self, action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(action) * self.action_limit

    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = self._squash_action(raw)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.sum(torch.log(1.0 - torch.tanh(raw).pow(2) + 1e-6), dim=-1)
        return action, log_prob, mean

    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(obs)
        return self._squash_action(mean)

    def log_prob(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        clipped = action / self.action_limit
        clipped = torch.clamp(clipped, -0.999999, 0.999999)
        raw = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        log_prob = dist.log_prob(raw).sum(dim=-1)
        log_prob -= torch.sum(torch.log(1.0 - clipped.pow(2) + 1e-6), dim=-1)
        return log_prob


class ValueMLP(nn.Module):
    def __init__(self, obs_dim: int, hidden: list[int]) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim, hidden, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PPOAgent:
    def __init__(self, cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.device = cfg.device

        self.actor = GaussianActor(
            cfg.obs_dim, cfg.action_dim, cfg.action_limit, cfg.actor_hidden
        ).to(self.device)
        self.critic = ValueMLP(cfg.obs_dim, cfg.critic_hidden).to(self.device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    def act(
        self, obs: np.ndarray, noise_sigma: float = 0.0, deterministic: bool = False
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(obs_t)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def get_action_and_value(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor.sample(obs_t)
            value = self.critic(obs_t)
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        values_old: torch.Tensor,
        epochs: int,
        batch_size: int,
    ) -> Dict[str, float]:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        data_size = obs.shape[0]
        actor_loss_val = 0.0
        critic_loss_val = 0.0
        entropy_val = 0.0
        approx_kl_val = 0.0

        stop_early = False
        for _ in range(epochs):
            indices = torch.randperm(data_size, device=self.device)
            for start in range(0, data_size, batch_size):
                idx = indices[start : start + batch_size]

                batch_obs = obs[idx]
                batch_actions = actions[idx]
                batch_logp_old = log_probs_old[idx]
                batch_returns = returns[idx]
                batch_adv = advantages[idx]
                batch_values_old = values_old[idx]

                logp = self.actor.log_prob(batch_obs, batch_actions)
                ratio = torch.exp(logp - batch_logp_old)
                approx_kl = (batch_logp_old - logp).mean()
                approx_kl_val = float(approx_kl.detach().cpu().item())
                if self.cfg.target_kl > 0 and approx_kl_val > self.cfg.target_kl:
                    stop_early = True
                    break
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value = self.critic(batch_obs)
                if self.cfg.clip_vloss:
                    value_clipped = batch_values_old + torch.clamp(
                        value - batch_values_old,
                        -self.cfg.clip_ratio,
                        self.cfg.clip_ratio,
                    )
                    value_loss_unclipped = (value - batch_returns).pow(2)
                    value_loss_clipped = (value_clipped - batch_returns).pow(2)
                    critic_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    critic_loss = F.mse_loss(value, batch_returns)

                mean, log_std = self.actor(batch_obs)
                dist = torch.distributions.Normal(mean, log_std.exp())
                entropy = dist.entropy().sum(dim=-1).mean()

                loss = (
                    actor_loss
                    + self.cfg.value_coef * critic_loss
                    - self.cfg.entropy_coef * entropy
                )

                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                if self.cfg.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()

                actor_loss_val = float(actor_loss.detach().cpu().item())
                critic_loss_val = float(critic_loss.detach().cpu().item())
                entropy_val = float(entropy.detach().cpu().item())
            if stop_early:
                break

        return {
            "actor_loss": actor_loss_val,
            "critic_loss": critic_loss_val,
            "entropy": entropy_val,
            "approx_kl": approx_kl_val,
        }

    def state_dict(self) -> Dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic_opt.load_state_dict(state["critic_opt"])
