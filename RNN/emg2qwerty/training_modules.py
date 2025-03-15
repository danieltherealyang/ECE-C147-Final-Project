# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
import math
import torch
from torch import nn

class SpectrogramNorm(nn.Module):
    """Applies 2D batch normalization over spectrograms per electrode channel per band."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels
        self.batch_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        assert self.channels == bands * C
        x = inputs.movedim(0, -1)  # (N, bands, C, freq, T)
        x = x.reshape(N, bands * C, freq, T)
        x = self.batch_norm(x)
        x = x.reshape(N, bands, C, freq, T)
        return x.movedim(-1, 0)  # (T, N, bands, C, freq)

class RotationInvariantMLP(nn.Module):
    """Applies an MLP over rotated versions of the input and pools the results."""
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()
        assert len(mlp_features) > 0
        mlp = []
        for out_features in mlp_features:
            mlp.extend([nn.Linear(in_features, out_features), nn.ReLU()])
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)
        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling
        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (T, N, C, ...)
        x = torch.stack([inputs.roll(offset, dims=2) for offset in self.offsets], dim=2)
        x = self.mlp(x.flatten(start_dim=3))
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)

class MultiBandRotationInvariantMLP(nn.Module):
    """Applies a separate RotationInvariantMLP per band for multi-band inputs."""
    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands
        inputs_per_band = inputs.unbind(self.stack_dim)
        outputs_per_band = [mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)]
        return torch.stack(outputs_per_band, dim=self.stack_dim)

class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per 'Sequence-to-Sequence Speech Recognition
    with Time-Depth Separable Convolutions' (Hannun et al.)."""
    def __init__(self, channels: int, width: int, kernel_width: int) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape  # TNC
        x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
        x = self.conv2d(x)
        x = self.relu(x)
        x = x.reshape(N, C, -1).movedim(-1, 0)  # (T, N, C)
        T_out = x.shape[0]
        x = x + inputs[-T_out:]
        return self.layer_norm(x)

class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block with skip connection and layer normalization."""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc_block = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Linear(num_features, num_features),
        )
        self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc_block(inputs)
        x = x + inputs
        return self.layer_norm(x)

class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing TDSConv2dBlock and
    TDSFullyConnectedBlock modules."""
    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
    ) -> None:
        super().__init__()
        assert len(block_channels) > 0
        tds_conv_blocks = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width),
                    TDSFullyConnectedBlock(num_features),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)

# --- New: Positional Encoding Module ---
class PositionalEncoding(nn.Module):
    """Injects positional information into a sequence. Expects input shape
    (seq_len, batch_size, d_model)."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
