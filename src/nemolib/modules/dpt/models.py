from typing import Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange

from .base_model import BaseModel
from .blocks import FeatureFusionBlock_custom, Interpolate, _make_scratch


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head=None,
        features=256,
        hooks: list[int] = [0, 1, 2, 3],
        layer_dims: list[int] = [768, 768, 768, 768],
        channels_last=False,
        use_bn=False,
        enable_attention_hooks=False,
    ):

        super(DPT, self).__init__()

        self.channels_last = channels_last

        self.hooks = hooks
        self.features = features

        self.act_1_prescale = nn.Sequential(
            nn.Conv2d(
                in_channels=layer_dims[0],
                out_channels=layer_dims[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=layer_dims[0],
                out_channels=layer_dims[0],
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.act_2_prescale = nn.Sequential(
            nn.Conv2d(
                in_channels=layer_dims[1],
                out_channels=layer_dims[1],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.ConvTranspose2d(
                in_channels=layer_dims[1],
                out_channels=layer_dims[1],
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
                dilation=1,
                groups=1,
            ),
        )

        self.act_3_prescale = nn.Sequential(
            nn.Conv2d(
                in_channels=layer_dims[2],
                out_channels=layer_dims[2],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )

        self.act_4_prescale = nn.Sequential(
            nn.Conv2d(
                in_channels=layer_dims[3],
                out_channels=layer_dims[3],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Conv2d(
                in_channels=layer_dims[3],
                out_channels=layer_dims[3],
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.scratch = _make_scratch(layer_dims, self.features, groups=1, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(self.features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(self.features, use_bn)

        # Delete unused conf for refinenet 4
        del self.scratch.refinenet4.resConfUnit1

        self.head = head if head is not None else nn.Identity()

    def forward(self, tokens: list[torch.Tensor]):
        layer_1, layer_2, layer_3, layer_4 = tokens[0], tokens[1], tokens[2], tokens[3]

        # Prescale tokens
        layer_1_sc = self.act_1_prescale(layer_1)
        layer_2_sc = self.act_2_prescale(layer_2)
        layer_3_sc = self.act_3_prescale(layer_3)
        layer_4_sc = self.act_4_prescale(layer_4)

        layer_1_rn = self.scratch.layer1_rn(layer_1_sc)
        layer_2_rn = self.scratch.layer2_rn(layer_2_sc)
        layer_3_rn = self.scratch.layer3_rn(layer_3_sc)
        layer_4_rn = self.scratch.layer4_rn(layer_4_sc)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.head(path_1)

        return out


class DPTPointMapConfidenceModel(DPT):
    def __init__(self, num_channels: int = 4, patch_size: Union[int, Tuple[int, int]] = 16, **kwargs):
        super().__init__()
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.num_channels = num_channels
        self.head = nn.Sequential(
            nn.Conv2d(self.features, self.features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(self.features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, self.num_channels, kernel_size=1, stride=1, padding=0),
        )
        self.patch_linear = nn.Linear(
            self.num_channels * 16 * 16, self.num_channels * self.patch_size[0] * self.patch_size[1], device="cuda"
        )

    def forward(self, tokens: list, image_size):
        out = super().forward(tokens)

        H, W = image_size
        # Adapt to arbitrary patch size
        out = nn.functional.unfold(out, (16, 16), stride=16)
        out = rearrange(out, "b kernel_size blocks -> b blocks kernel_size")
        out = self.patch_linear(out)
        out = rearrange(out, "b blocks kernel_size -> b kernel_size blocks ")
        out = nn.functional.fold(out, image_size, self.patch_size, stride=self.patch_size)

        return out
