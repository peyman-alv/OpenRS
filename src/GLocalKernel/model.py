from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalLocalKernelModel(nn.Module):
    def __init__(
        self,
        local_kernel_net: nn.Module,
        gk_size: int,
        dot_scale: float,
        n_m: int,
    ) -> None:
        super(GlobalLocalKernelModel, self).__init__()

        self.local_kernel_net = local_kernel_net
        self.gk_size = gk_size
        self.dot_scale = dot_scale

        self.conv_kernel = nn.Parameter(torch.randn(n_m, gk_size**2) * 0.1)
        nn.init.xavier_uniform_(
            self.conv_kernel, gain=torch.nn.init.calculate_gain("relu")
        )

    def forward(
        self, x: torch.Tensor, x_local: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gk = self.global_kernel(x_local)
        x = self.global_conv(x, gk)
        x, reg_loss = self.local_kernel_net(x)
        return x, reg_loss

    def global_kernel(self, x: torch.Tensor) -> torch.Tensor:
        avg_pooling = torch.mean(x, dim=1).view(1, -1)
        gk = torch.matmul(avg_pooling, self.conv_kernel) * self.dot_scale
        gk = gk.view(1, 1, self.gk_size, self.gk_size)
        return gk

    def global_conv(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(0).unsqueeze(0)
        conv2d = nn.LeakyReLU()(F.conv2d(x, W, stride=1, padding=1))
        conv2d = conv2d.squeeze(0).squeeze(0)
        return conv2d
