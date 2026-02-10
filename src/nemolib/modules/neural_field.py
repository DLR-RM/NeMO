"""Neural Field network for Distance function."""

from typing import Callable, Optional

import torch
import torch.nn as nn

from .mlp import SkipMLP
from .positional_encoding import PositionalEncoding3D


class NeuralField(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_layers: int,
        neurons_per_layer: int,
        pos_enc_length: Optional[int] = None,
        output_function: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        **kwargs,
    ) -> None:
        super().__init__()
        if pos_enc_length is not None:
            final_input_dimension = pos_enc_length
            self.pos_emb_layer = PositionalEncoding3D(pos_enc_length)
        else:
            final_input_dimension = input_dimension
            self.pos_emb_layer = nn.Identity()

        self.skip_mlp = SkipMLP(
            final_input_dimension,
            output_dimension,
            hidden_layers,
            neurons_per_layer,
            res_connection=2,
            activation_function_name="gelu",
            **kwargs,
        )

        self.output_function = output_function

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.pos_emb_layer(input)
        x = self.skip_mlp(x)
        x = self.output_function(x)  # for unsigned distance, it's best to use torch.tanh
        return x
