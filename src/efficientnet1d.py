"""
Modified from the original implementation by bilzard
https://github.com/bilzard/kaggle-hms-bilzard/blob/main/src/model/eeg_encoder/efficientnet1d.py
"""

import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch import Tensor
import torch.nn.functional as F
import torch

from .transformer_block import MultiHeadSelfAttention


class GeMPool1d(nn.Module):
    def __init__(self, p: int = 3, eps: float = 1e-4):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: b c t
        """
        return F.adaptive_avg_pool1d(x.clamp(min=self.eps).pow(self.p), 1).pow(
            1.0 / self.p
        )


class SqueezeExcite(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        se_ratio: int,
        activation: type[nn.Module] = nn.ELU,
    ):
        super().__init__()

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(hidden_dim, hidden_dim // se_ratio, kernel_size=1),
            activation(inplace=True),
            nn.Conv2d(hidden_dim // se_ratio, hidden_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x)


class ConvBnAct2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, ...] = (1, 1),
        stride: tuple[int, ...] = (1, 1),
        padding: tuple[int, ...] | str = "same",
        groups: int = 1,
        activation: type[nn.Module] = nn.ELU,
        drop_path_rate: float = 0.0,
        skip: bool = False,
    ):
        super().__init__()
        assert (
            kernel_size >= stride
        ), f"kernel_size must be greater than stride. Got {kernel_size} and {stride}."

        match padding:
            case "valid":
                padding = (0, 0)
            case "same":
                padding = tuple([(k - s) // 2 for k, s in zip(kernel_size, stride)])
            case _:
                pass

        self.has_skip = skip and stride == 1 and in_channels == out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,  # type: ignore
                stride,  # type: ignore
                padding,  # type: ignore
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True) if activation is not None else nn.Identity(),
        )
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut

        return x


class DepthWiseSeparableConv(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        kernel_size: tuple[int, ...],
        activation: type[nn.Module],
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
    ):
        super().__init__()
        self.has_skip = skip
        modules: list[nn.Module] = [
            ConvBnAct2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            )
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
            )
        modules.append(ConvBnAct2d(hidden_dim, hidden_dim, activation=activation))
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
            )

        self.conv = nn.Sequential(*modules)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.conv(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        depth_multiplier: int,
        kernel_size: tuple[int, ...],
        activation: type[nn.Module],
        se_ratio: int = 4,
        skip: bool = True,
        drop_path_rate: float = 0.0,
        se_after_dw_conv: bool = False,
    ):
        super().__init__()
        self.has_skip = skip

        modules: list[nn.Module] = [
            ConvBnAct2d(in_channels, hidden_dim, activation=activation),
            ConvBnAct2d(
                hidden_dim,
                hidden_dim * depth_multiplier,
                kernel_size=kernel_size,
                groups=hidden_dim,
                activation=activation,
            ),
        ]
        if se_after_dw_conv:
            modules.append(
                SqueezeExcite(
                    hidden_dim * depth_multiplier,
                    se_ratio=se_ratio,
                    activation=activation,
                )
            )
        modules.append(
            ConvBnAct2d(
                hidden_dim * depth_multiplier,
                hidden_dim,
                activation=activation,
            ),
        )
        if not se_after_dw_conv:
            modules.append(
                SqueezeExcite(hidden_dim, se_ratio=se_ratio, activation=activation)
            )

        self.inv_res = nn.Sequential(*modules)
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        if self.has_skip and in_channels != hidden_dim:
            self.skip_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=(1, 1))

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.inv_res(x)
        if self.has_skip:
            if self.in_channels != self.hidden_dim:
                shortcut = self.skip_conv(shortcut)

            x = self.drop_path(x) + shortcut
        return x


class ResBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer: nn.Module,
        pool_size: int = 2,
        skip: bool = True,
    ):
        super().__init__()
        self.has_skip = skip

        if self.has_skip:
            self.pool = nn.Conv2d(in_channels, out_channels, kernel_size=(1, pool_size), stride=(1, pool_size))

        self.layer = nn.Sequential(
            layer,
            nn.MaxPool2d(kernel_size=(1, pool_size), stride=(1, pool_size)),
        )

    def forward(self, x: Tensor) -> Tensor:
        if not self.has_skip:
            return self.layer(x)

        return self.pool(x) + self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dims)
        # self.mhsa = SharedQkMultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)
        self.mhsa = MultiHeadSelfAttention(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mhsa(self.norm(x))
        return x


class TransformerChannelMixer(nn.Module):
    def __init__(self, hidden_dims: int, num_heads: int, **kwargs):
        super().__init__()
        self.transformer = TransformerBlock(hidden_dims, num_heads, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        T = x.shape[-1]
        x = rearrange(x, "b c ch t -> b (t ch) c")
        x = self.transformer(x)
        x = rearrange(x, "b (t ch) c -> b c ch t", t=T)
        return x


class EfficientNet1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        depth_multiplier: int = 4,
        stem_kernel_size: int = 3,
        stem_stride: int = 1,
        kernel_sizes: list[int] = [3, 3, 3, 5, 5],
        pool_sizes: list[int] = [2, 2, 2, 2, 2],
        layers: list[int] | int = [1, 2, 2, 3, 3],
        skip_in_block: bool = True,
        skip_in_layer: bool = True,
        activation=nn.ELU,
        drop_path_rate: float = 0.0,
        use_ds_conv: bool = True,
        se_after_dw_conv: bool = False,
        use_channel_mixer: bool = False,
        channel_mixer_kernel_size: int = 3,
        mixer_type: str = "sc",
        transformer_merge_type: str = "add",
    ):
        super().__init__()
        if isinstance(layers, int):
            layers = [layers] * len(kernel_sizes)

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth_multiplier = depth_multiplier
        self.temporal_pool_sizes = pool_sizes
        self.temporal_layers = layers
        self.num_eeg_channels = in_channels // 2
        self.drop_path_rate = drop_path_rate

        self.stem_conv = ConvBnAct2d(
            in_channels,
            hidden_dim,
            kernel_size=(1, stem_kernel_size),
            activation=activation,
            drop_path_rate=drop_path_rate,
            stride=(1, stem_stride),
        )

        self.efficient_net = nn.Sequential(
            *[
                ResBlock2d(
                    in_channels=self.layer_num_to_channel(i),
                    out_channels=self.layer_num_to_channel(i + 1),
                    layer=nn.Sequential(
                        *[
                            DepthWiseSeparableConv(
                                hidden_dim=hidden_dim * 2 ** (i - 1),
                                kernel_size=(1, k),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            if i == 0 and use_ds_conv
                            else InvertedResidual(
                                in_channels=self.layer_num_to_channel(i) if ii == 0 else self.layer_num_to_channel(i + 1),
                                hidden_dim=self.layer_num_to_channel(i + 1),
                                depth_multiplier=depth_multiplier,
                                kernel_size=(1, k),
                                activation=activation,
                                se_ratio=depth_multiplier,
                                drop_path_rate=drop_path_rate,
                                skip=skip_in_layer,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            for ii in range(nl)
                        ],
                        (
                            InvertedResidual(
                                hidden_dim=self.layer_num_to_channel(i + 1),
                                kernel_size=(channel_mixer_kernel_size, 1),
                                activation=activation,
                                depth_multiplier=depth_multiplier,
                                se_ratio=depth_multiplier,
                                se_after_dw_conv=se_after_dw_conv,
                            )
                            if mixer_type == "ir"
                            else (
                                DepthWiseSeparableConv(
                                    hidden_dim=self.layer_num_to_channel(i + 1),
                                    kernel_size=(channel_mixer_kernel_size, 1),
                                    activation=activation,
                                    se_ratio=depth_multiplier,
                                    se_after_dw_conv=se_after_dw_conv,
                                )
                                if mixer_type == "sc"
                                else TransformerChannelMixer(
                                    self.layer_num_to_channel(i + 1),
                                    num_heads=depth_multiplier,
                                    merge_type=transformer_merge_type,
                                )
                            )
                        )
                        if use_channel_mixer
                        else nn.Identity(),
                    ),
                    pool_size=p,
                    skip=skip_in_block,
                )
                for i, (k, p, nl) in enumerate(zip(kernel_sizes, pool_sizes, layers))
            ]
        )

    def layer_num_to_channel(self, i: int) -> int:
        hidden_dim = self.hidden_dim
        if i % 2 == 0:
            return int(hidden_dim * 2 ** (i // 2))
        else:
            return int(hidden_dim * 2 ** ((i - 1) // 2) * 1.5)

    @property
    def out_channels(self) -> int:
        return self.layer_num_to_channel(len(self.temporal_layers))

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem_conv(x)
        x = self.efficient_net(x)  # b c ch t = 2, 64, 4, 16
        return x


if __name__ == "__main__":
    kwargs = {
        "in_channels": 4,
        "hidden_dim": 32,
        "depth_multiplier": 4,
        "stem_kernel_size": 3,
        "kernel_sizes": [3, 3, 5, 5, 3],
        "pool_sizes": [2, 2, 2, 2, 2],
        "layers": 3,
        "skip_in_block": True,
        "skip_in_layer": True,
        "drop_path_rate": 0.0,
        "use_ds_conv": False,
        "se_after_dw_conv": False,
        "use_channel_mixer": True,
        "channel_mixer_kernel_size": 3,
        "mixer_type": "sc"
    }
    model = EfficientNet1d(**kwargs)
    x = torch.randn(2, 4, 4, 512)
    y = model(x)
    print(y.shape)
