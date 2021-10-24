from torch import nn
import torch.nn.functional as F
import numpy as np


class DecodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodingBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv3d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv3d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.upsamle = F.interpolate
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        size = tuple(2 * np.array(x.shape[-3:]))
        out = self.backbone(x)
        out = self.upsamle(out, size=size, mode='trilinear')
        identity = self.upsamle(x, size=size, mode='trilinear')
        out += self.shortcut(identity)
        return F.leaky_relu(out)


class EncodingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodingBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 2, 1, groups=in_channels),
            nn.Conv3d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv3d(out_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels)
        )
        self.shortcut = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.backbone(x)
        identity = F.avg_pool2d(x, 2)
        out += self.shortcut(identity)
        return F.leaky_relu(out)


class Encoder(nn.Sequential):
    def __init__(self, in_ch: int = 1, num_blocks: int = 5):
        super(Encoder, self).__init__(*[EncodingBlock(in_ch * 2 ** i, in_ch * 2 ** (i + 1)) for i in range(num_blocks)])


class Decoder(nn.Sequential):
    def __init__(self, in_ch: int = 32, num_blocks: int = 5):
        super(Decoder, self).__init__(*[DecodingBlock(in_ch // (2 ** i), in_ch // (2 ** (i + 1)))
                                        for i in range(num_blocks)])


class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int = 1, num_encoding_blocks: int = 5,
                 out_channels_first_layer: int = 64, **kwargs):
        super(AutoEncoder, self).__init__()
        in_ch, out_ch, n = in_channels, out_channels_first_layer, num_encoding_blocks
        self.features = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )
        self.encoder = Encoder(out_ch, n)
        self.decoder = Decoder(out_ch * 2 ** n, n)
        self.head = nn.Sequential(
            nn.Conv3d(out_ch, in_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.head(x)
        return x
