# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .dpt_head import create_dpt_head
from .linear_head import create_linear_head


def head_factory(head_type, output_mode, config):
    """ " build a prediction head for the decoder"""
    if head_type == "linear" and output_mode == "pts3d":
        return create_linear_head(config)
    elif head_type == "dpt" and output_mode == "pts3d":
        return create_dpt_head(config)
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")
