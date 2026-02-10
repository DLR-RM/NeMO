# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

from .postprocess import postprocess


def create_linear_head(config, has_conf=False):
    depth_mode = ["tanh", -1, 1]  # Use tanh to map output to -1 and 1
    conf_mode = ["tanh", 1, 2]
    return LinearPts3d(
        config.model.linear_head_dimension, config.model.patch_size, depth_mode, conf_mode, has_conf, postprocess=postprocess
    )


class LinearPts3d(nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, feature_dim, patch_size, depth_mode, conf_mode, has_conf=False, postprocess=None):
        super().__init__()
        self.patch_size = patch_size
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.has_conf = has_conf
        self.postprocess = postprocess

        self.proj = nn.Linear(feature_dim, (3 + has_conf) * self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H // self.patch_size, W // self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        if self.postprocess is not None:
            return self.postprocess(feat, self.depth_mode, self.conf_mode)
        else:
            return feat
