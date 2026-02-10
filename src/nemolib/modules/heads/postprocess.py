# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# --------------------------------------------------------
import torch
import torch.nn.functional as F


def postprocess(out, depth_mode, conf_mode):
    """
    extract 3D points/confidence from prediction head output
    """
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,3
    out_channels = fmap.shape[-1]
    res = {}
    if out_channels < 3:
        raise RuntimeError(f"Output channel has to be at least 3, but got `{out_channels}`")
    if out_channels >= 3:
        res["pts3d"] = reg_dense_pts3d(fmap[:, :, :, 0:3], mode=depth_mode)
    if out_channels >= 4:
        res["conf"] = reg_dense_conf(fmap[:, :, :, 3], mode=conf_mode)
    if out_channels >= 5:
        res["depth_scaled"] = reg_dense_depth_scaled(fmap[:, :, :, 4], mode=("softplus", 0, 3))  # logits
    if out_channels >= 6:
        res["depth_scaled_conf"] = reg_dense_conf(fmap[:, :, :, 5], mode=conf_mode)
    if out_channels >= 7:
        res["mask"] = torch.sigmoid(fmap[:, :, :, 6])  # logits
    if out_channels >= 8:
        res["mask_full"] = torch.sigmoid(fmap[:, :, :, 7])  # logits
    return res


def reg_dense_depth_scaled(depth, mode):
    """Extract scaled depth from head output"""
    mode, vmin, vmax = mode
    if mode == "softplus":
        return torch.clamp(F.softplus(depth), vmin, vmax)
    raise ValueError(f"bad {mode=}")


def reg_dense_pts3d(xyz, mode):
    """
    extract 3D points from prediction head output
    """
    mode, vmin, vmax = mode

    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))

    if mode == "tanh":
        if no_bounds:
            return torch.tanh(xyz)
        return torch.tanh(xyz).clip(min=vmin, max=vmax)

    if mode == "linear":
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == "square":
        return xyz * d.square()

    if mode == "exp":
        return xyz * torch.expm1(d)

    raise ValueError(f"bad {mode=}")


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if vmin == "inf":
        vmin = float("inf")
    if vmax == "inf":
        vmax = float("inf")
    if mode == "tanh":
        return torch.clip(vmin + (0.5 * (1 + torch.tanh(x))), max=vmax)  # Maps between 1 and 2
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"bad {mode=}")
