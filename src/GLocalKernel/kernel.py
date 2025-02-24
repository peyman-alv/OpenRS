"""
PyTorch implementation of the G-Local Kernel Network (GLocalKernel) model.

official repository: https://github.com/usydnlp/Glocal_K/tree/main
official paper: https://arxiv.org/pdf/2108.12184
"""

from typing import Tuple

import torch
import torch.nn as nn


class KernelLayer(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_hid: int,
        n_dim: int,
        lambda_s: float,
        lambda_2: float,
        activation: nn.Module,
    ) -> None:
        super(KernelLayer, self).__init__()
        self.n_in = n_in
        self.n_hid = n_hid
        self.n_dim = n_dim
        self.lambda_s = lambda_s
        self.lambda_2 = lambda_2
        self.activation = activation

        self.W = nn.Parameter(torch.Tensor(n_in, n_hid))
        self.u = nn.Parameter(torch.Tensor(n_in, 1, n_dim))
        self.v = nn.Parameter(torch.Tensor(1, n_hid, n_dim))
        self.b = nn.Parameter(torch.Tensor(n_hid))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W)
        nn.init.normal_(self.u, mean=0.0, std=1e-3)
        nn.init.normal_(self.v, mean=0.0, std=1e-3)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w_hat = self.compute_kernel(self.u, self.v)
        W_eff = self.W * w_hat
        y = torch.matmul(x, W_eff) + self.b
        y = self.activation(y)

        sparse_reg_term = self.lambda_s * torch.sum(w_hat**2)
        l2_reg_term = self.lambda_2 * torch.sum(self.W**2)
        reg_term = sparse_reg_term + l2_reg_term

        return y, reg_term

    def compute_kernel(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        dist = torch.norm(u - v, p=2, dim=2)
        hat = torch.clamp(1 - dist**2, min=0.0)
        return hat


class KernelNet(nn.Module):
    def __init__(
        self,
        n_u: int,
        n_layers: int,
        n_hid: int,
        n_dim: int,
        lambda_s: float,
        lambda_2: float,
        activation: nn.Module,
    ):
        super(KernelNet, self).__init__()

        self.layers = nn.ModuleList(
            [KernelLayer(n_u, n_hid, n_dim, lambda_s, lambda_2, activation)]
            + [
                KernelLayer(n_hid, n_hid, n_dim, lambda_s, lambda_2, activation)
                for _ in range(n_layers - 1)
            ]
            + [
                KernelLayer(
                    n_hid, n_u, n_dim, lambda_s, lambda_2, activation=nn.Identity()
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total_reg_term = 0.0
        for layer in self.layers:
            x, reg = layer(x)
            total_reg_term += reg
        return x, total_reg_term
