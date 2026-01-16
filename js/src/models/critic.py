import torch
import torch.nn as nn

from .actor import build_mlp


class CriticMLP(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.net = build_mlp(obs_dim + action_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q = self.net(torch.cat([obs, action], dim=-1))
        return q.squeeze(-1)


class CriticLSTM(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_size: int,
        num_layers: int,
        mlp_hidden: list[int],
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=obs_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.mlp = build_mlp(hidden_size + action_dim, mlp_hidden, 1)

    def forward(self, obs_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # obs_seq: (batch, seq_len, obs_dim)
        _, (h_n, _) = self.lstm(obs_seq)
        last_hidden = h_n[-1]
        q = self.mlp(torch.cat([last_hidden, action], dim=-1))
        return q.squeeze(-1)
