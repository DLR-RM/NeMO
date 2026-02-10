"""Neural Field network for Distance function."""

import torch
import torch.nn as nn


class SkipMLP(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        hidden_layers: int,
        neurons_per_layer: int,
        res_connection: int = 2,
        activation_function_name: str = "gelu",
    ) -> None:
        super().__init__()
        self.res_connection = res_connection

        if activation_function_name == "gelu":
            self.activation_function = nn.GELU()
        elif activation_function_name == "relu":
            self.activation_function = nn.ReLU()
        else:
            raise NotImplementedError(f"Activation function not implemented {activation_function_name=}")

        self.input_layer = nn.Linear(input_dimension, neurons_per_layer)

        linear_modules = nn.ModuleList()
        for _ in range(hidden_layers):
            linear_modules.extend(
                [
                    nn.Sequential(
                        nn.Linear(neurons_per_layer, neurons_per_layer),
                        nn.LayerNorm(neurons_per_layer),
                        self.activation_function,
                    )
                ]
            )

        self.hidden_layers = linear_modules  # nn.Sequential(*linear_relu_modules)

        self.output_layer = nn.Linear(neurons_per_layer, output_dimension)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.activation_function(self.input_layer(input))
        residual = x
        for n, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if n != 0 and n % self.res_connection == 0:
                x = x + residual
        x = self.output_layer(x)
        return x
