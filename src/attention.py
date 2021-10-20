import torch
from torch import nn
import torch.nn.functional as F
from unet import UNet
from unet.decoding import Decoder, DecodingBlock
from typing import Optional


CHANNELS_DIMENSION = 1


class AttentionUnet(UNet):
    def __init__(self,
                 in_channels: int = 1,
                 out_classes: int = 2,
                 dimensions: int = 2,
                 num_encoding_blocks: int = 5,
                 out_channels_first_layer: int = 64,
                 normalization: Optional[str] = None,
                 pooling_type: str = 'max',
                 upsampling_type: str = 'conv',
                 preactivation: bool = False,
                 residual: bool = False,
                 padding: int = 0,
                 padding_mode: str = 'zeros',
                 activation: Optional[str] = 'ReLU',
                 initial_dilation: Optional[int] = None,
                 dropout: float = 0,
                 monte_carlo_dropout: float = 0
                 ):
        super().__init__(in_channels, out_classes, dimensions, num_encoding_blocks, out_channels_first_layer,
                         normalization, pooling_type, upsampling_type, preactivation, residual, padding, padding_mode,
                         activation, initial_dilation, dropout, monte_carlo_dropout)
        depth = num_encoding_blocks - 1

        if dimensions == 2:
            power = depth - 1
        elif dimensions == 3:
            power = depth

        in_channels_skip_connection = out_channels_first_layer * 2 ** power
        num_decoding_blocks = depth
        self.decoder = AttentionDecoder(
            in_channels_skip_connection,
            dimensions,
            upsampling_type,
            num_decoding_blocks,
            normalization=normalization,
            preactivation=preactivation,
            residual=residual,
            padding=padding,
            padding_mode=padding_mode,
            activation=activation,
            initial_dilation=self.encoder.dilation,
            dropout=dropout,
        )


class AttentionDecoder(Decoder):
    def __init__(
            self,
            in_channels_skip_connection: int,
            dimensions: int,
            upsampling_type: str,
            num_decoding_blocks: int,
            normalization: Optional[str],
            preactivation: bool = False,
            residual: bool = False,
            padding: int = 0,
            padding_mode: str = 'zeros',
            activation: Optional[str] = 'ReLU',
            initial_dilation: Optional[int] = None,
            dropout: float = 0,
            ):
        super().__init__(in_channels_skip_connection, dimensions, upsampling_type, num_decoding_blocks, normalization,
                         preactivation, residual, padding, padding_mode, activation, initial_dilation, dropout)
        upsampling_type = fix_upsampling_type(upsampling_type, dimensions)
        self.decoding_blocks = nn.ModuleList()
        self.dilation = initial_dilation
        for _ in range(num_decoding_blocks):
            decoding_block = AttentionDecodingBlock(
                in_channels_skip_connection,
                dimensions,
                upsampling_type,
                normalization=normalization,
                preactivation=preactivation,
                residual=residual,
                padding=padding,
                padding_mode=padding_mode,
                activation=activation,
                dilation=self.dilation,
                dropout=dropout,
            )
            self.decoding_blocks.append(decoding_block)
            in_channels_skip_connection //= 2
            if self.dilation is not None:
                self.dilation //= 2


class AttentionDecodingBlock(DecodingBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_ch = args[0]
        self.attention = AttentionGate(n_ch, 2 * n_ch, n_ch)

    def forward(self, skip_connection, x):
        _x = x.clone()
        x = self.upsample(x)
        skip_connection = self.center_crop(skip_connection, x)
        attention = self.attention(skip_connection, _x)
        x = torch.cat((attention, x), dim=CHANNELS_DIMENSION)
        if self.residual:
            connection = self.conv_residual(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x += connection
        else:
            x = self.conv1(x)
            x = self.conv2(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels):
        super().__init__()
        self.theta = nn.Conv3d(in_channels=in_channels, out_channels=inter_channels,
                               kernel_size=2, stride=2, padding=0, bias=False)
        self.phi = nn.Conv3d(in_channels=gating_channels, out_channels=inter_channels,
                             kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv3d(in_channels=inter_channels, out_channels=1, kernel_size=1,
                             stride=1, padding=0, bias=True)

    def forward(self, x, g):
        batch_size = x.shape[0]
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        f = torch.relu(theta_x + phi_g)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = torch.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])
        sigm_psi_f = F.interpolate(sigm_psi_f, size=x.shape[2:], mode='trilinear')
        y = sigm_psi_f.expand_as(x) * x

        return y


def fix_upsampling_type(upsampling_type: str, dimensions: int):
    if upsampling_type == 'linear':
        if dimensions == 2:
            upsampling_type = 'bilinear'
        elif dimensions == 3:
            upsampling_type = 'trilinear'
    return upsampling_type
