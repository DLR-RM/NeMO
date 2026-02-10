"""
Encoding functions
"""

import torch
from torch import nn


class PositionalEncoding3D(nn.Module):
    def __init__(self, encoding_length: int = 36):
        super().__init__()
        assert encoding_length % 6 == 0, "Encoding length must be divisible by 6"
        self.encoding_length = encoding_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d_per_coord = self.encoding_length // 3
        device = x.device
        i = torch.arange(d_per_coord // 2, device=device)

        # NeRF-style frequency calculation
        freq = 2**i

        # Encode each coordinate separately
        encodings = []
        for coord in [x[..., 0], x[..., 1], x[..., 2]]:
            angles = coord[..., None] * freq
            enc = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            encodings.append(enc)

        return torch.cat(encodings, dim=-1)