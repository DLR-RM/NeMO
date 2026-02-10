# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------

import torch
import torch.nn as nn
from einops import rearrange

from ..dpt.models import DPTPointMapConfidenceModel

from .dpt_block import DPTOutputAdapter
from .postprocess import postprocess

inf = float("inf")


class DPTOutputAdapter_fix(DPTOutputAdapter):
    """
    Adapt croco's DPTOutputAdapter implementation for dust3r:
    remove duplicated weigths, and fix forward for dust3r
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # these are duplicated weights
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess
        self.patch_linear = torch.nn.Linear(
            self.num_channels * 16 * 16, self.num_channels * self.patch_size[0] * self.patch_size[1], device="cuda"
        )

    def forward(self, encoder_tokens: list[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, "Need to call init(dim_tokens_enc) function first"
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # Extract only task-relevant tokens and ignore global tokens.
        # layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers]

        # layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)  # [b, channels, multiple of 16, multiple of 16]

        # Adapt to arbitrary patch size
        out = torch.nn.functional.unfold(out, (16, 16), stride=16)
        out = rearrange(out, "b kernel_size blocks -> b blocks kernel_size")
        out = self.patch_linear(out)
        out = rearrange(out, "b blocks kernel_size -> b kernel_size blocks ")
        out = torch.nn.functional.fold(out, image_size, self.patch_size, stride=self.patch_size)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """DPT module for dust3r, can return 3D points + confidence for all pixels"""

    def __init__(
        self,
        *,
        hooks_idx=None,
        num_channels=1,
        postprocess=None,
        depth_mode=None,
        conf_mode=None,
        **kwargs,
    ):
        super().__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        dpt_args = dict(num_channels=num_channels, **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTPointMapConfidenceModel(**dpt_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(config):
    """
    return PixelwiseTaskWithDPT for given net params
    """
    # assert config.dec_depth > 9
    l2 = config.model.dpt_expected_depth
    feature_dim = config.model.dpt_feature_dim
    last_dim = feature_dim // 2
    out_nchan = config.model.dpt_output_channel
    ed = config.model.dpt_start_dim
    dd = config.model.dpt_end_dim
    depth_mode = config.model.dpt_depth_mode  # ["tanh", -1, 1]  # Use tanh to map output to -1 and 1
    conf_mode = config.model.dpt_conf_mode  # ["tanh", 1, 2]
    layer_dims = [config.model.dpt_feature_dim] * 4
    return PixelwiseTaskWithDPT(
        num_channels=out_nchan,
        patch_size=config.model.patch_size,
        feature_dim=feature_dim,
        last_dim=last_dim,
        hooks_idx=[
            int((l2 * (1 / 4)) - 1),
            int((l2 * (2 / 4)) - 1),
            int((l2 * (3 / 4)) - 1),
            int((l2 * (4 / 4)) - 1),
        ],
        dim_tokens=[ed, dd, dd, dd],
        postprocess=postprocess,
        depth_mode=depth_mode,
        conf_mode=conf_mode,
        head_type="regression",
        layer_dims=layer_dims,
    )
