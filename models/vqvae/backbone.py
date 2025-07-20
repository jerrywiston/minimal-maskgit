import torch
from torch import nn, einsum
import torch.nn.functional as F
import math

#################### Simple Implementation (for debug) ####################

class ResBlock(nn.Module):
    def __init__(self, input_channels, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, input_channels, 1),
        )

    def forward(self, x):
        out = self.conv(x)
        out += x
        out = F.relu(out)
        return out

class EncoderSimple(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=3, ch=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 2*ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*ch, 2*ch, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*ch, 2*ch//4),
            ResBlock(2*ch, 2*ch//4),
        )
        self.proj = nn.Conv2d(2*ch, embedding_dim, 1)

    def forward(self, x):
        h = self.net(x)
        z = self.proj(h)
        return z

class DecoderSimple(nn.Module):
    def __init__(self, embedding_dim=32, ch=64, output_channels=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, 2*ch, 3, padding=1),
            nn.ReLU(),
            ResBlock(2*ch, 2*ch//4),
            ResBlock(2*ch, 2*ch//4),
            nn.ConvTranspose2d(2*ch, ch, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ch, output_channels, 4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.net(x)

#################### Basic Bacone Implementation ####################

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

# Swish Function 
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# use GN for norm layer
def norm_layer(channels):
    return nn.GroupNorm(32, channels)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            norm_layer(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        `x` has shape `[batch_size, in_dim, height, width]`
        `t` has shape `[batch_size, time_dim]`
        """
        h = self.conv1(x)
        # Add time step embeddings
        h = self.conv2(h)
        return h + self.shortcut(x)

# Attention block with shortcut
class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0

        self.norm = norm_layer(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))
        q, k, v = qkv.reshape(B * self.num_heads, -1, H * W).chunk(3, dim=1)
        scale = 1. / math.sqrt(math.sqrt(C // self.num_heads))
        attn = torch.einsum("bct,bcs->bts", q * scale, k * scale)
        attn = attn.softmax(dim=-1)
        h = torch.einsum("bts,bcs->bct", attn, v)
        h = h.reshape(B, -1, H, W)
        h = self.proj(h)
        return h + x

############################################################

class Encoder(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=3, ch=64, downsample_steps=2, max_channel_scale=4):
        super().__init__()
        # Stem
        self.net = nn.ModuleList([nn.Conv2d(input_channels, ch, kernel_size=3, padding=1)])
        # Downsample Blocks
        in_scale, out_scale = 0, 0
        for i in range(downsample_steps):
            in_scale = 2**i
            if in_scale > max_channel_scale:
                in_scale = 4
            out_scale = 2**(i+1)
            if out_scale > max_channel_scale:
                out_scale = 4
            self.net.append(ResidualBlock(in_scale*ch, out_scale*ch))
            self.net.append(Downsample(out_scale*ch, with_conv=True))
        # Output Blocks
        self.net.append(ResidualBlock(out_scale*ch, out_scale*ch))
        self.net.append(ResidualBlock(out_scale*ch, out_scale*ch))
        self.net.append(nn.Conv2d(out_scale*ch, embedding_dim, kernel_size=3, padding=1))

    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h)
        return h

class Decoder(nn.Module):
    def __init__(self, embedding_dim=32, output_channels=3, ch=64, downsample_steps=2, max_channel_scale=4):
        super().__init__()
        scale = 2**downsample_steps
        if scale > max_channel_scale:
            scale = max_channel_scale
        # Stem
        self.net = nn.ModuleList([nn.Conv2d(embedding_dim, scale*ch, kernel_size=3, padding=1)])
        # Input Blocks
        self.net.append(ResidualBlock(scale*ch, scale*ch))
        self.net.append(ResidualBlock(scale*ch, scale*ch))
        # Upsample Blocks
        in_scale, out_scale = 0, 0
        for i in reversed(range(downsample_steps)):
            in_scale = 2**(i+1)
            if in_scale > max_channel_scale:
                in_scale = max_channel_scale
            out_scale = 2**i
            if out_scale > max_channel_scale:
                out_scale = max_channel_scale
            self.net.append(nn.Upsample(scale_factor=(2,2)))
            self.net.append(ResidualBlock(in_scale*ch, out_scale*ch))
        self.net.append(nn.Conv2d(out_scale*ch, output_channels, kernel_size=3, padding=1))

    def forward(self, z):
        h = z
        for layer in self.net:
            h = layer(h)
        return h

## 

class EncoderAtt(nn.Module):
    def __init__(self, input_channels=3, embedding_dim=8, ch=128, downsample_steps=2, num_res_blocks=2, max_channel_scale=4):
        super().__init__()
        # Stem
        self.net = nn.ModuleList([nn.Conv2d(input_channels, ch, kernel_size=3, padding=1)])
        # Downsample Blocks
        in_scale, out_scale = 0, 0
        for i in range(downsample_steps):
            in_scale = 2**i
            if in_scale > max_channel_scale:
                in_scale = 4
            out_scale = 2**(i+1)
            if out_scale > max_channel_scale:
                out_scale = 4
            for j in range(num_res_blocks):
                if j == num_res_blocks-1:
                    self.net.append(ResidualBlock(in_scale*ch, out_scale*ch))
                else:
                    self.net.append(ResidualBlock(in_scale*ch, in_scale*ch))
            self.net.append(Downsample(out_scale*ch, with_conv=True))
        # Middle Blocks
        self.net.append(ResidualBlock(out_scale*ch, out_scale*ch))
        self.net.append(AttentionBlock(out_scale*ch))
        self.net.append(ResidualBlock(out_scale*ch, out_scale*ch))
        # Output
        self.net.append(nn.GroupNorm(32, out_scale*ch))
        self.net.append(Swish())
        self.net.append(nn.Conv2d(out_scale*ch, embedding_dim, kernel_size=3, padding=1))

    def forward(self, x):
        h = x
        for layer in self.net:
            h = layer(h)
        return h

class DecoderAtt(nn.Module):
    def __init__(self, embedding_dim=8, output_channels=3, ch=128, downsample_steps=2, num_res_blocks=2, max_channel_scale=4):
        super().__init__()
        scale = 2**downsample_steps
        if scale > max_channel_scale:
            scale = max_channel_scale
        # Stem
        self.net = nn.ModuleList([nn.Conv2d(embedding_dim, scale*ch, kernel_size=3, padding=1)])
        # Middle Blocks
        self.net.append(ResidualBlock(scale*ch, scale*ch))
        self.net.append(AttentionBlock(scale*ch))
        self.net.append(ResidualBlock(scale*ch, scale*ch))
        # Upsample Blocks
        in_scale, out_scale = 0, 0
        for i in reversed(range(downsample_steps)):
            in_scale = 2**(i+1)
            if in_scale > max_channel_scale:
                in_scale = max_channel_scale
            out_scale = 2**i
            if out_scale > max_channel_scale:
                out_scale = max_channel_scale
            self.net.append(nn.Upsample(scale_factor=(2,2)))
            for j in range(num_res_blocks):
                if j == num_res_blocks-1:
                    self.net.append(ResidualBlock(in_scale*ch, out_scale*ch))
                else:
                    self.net.append(ResidualBlock(in_scale*ch, in_scale*ch))
        # Output
        self.net.append(nn.GroupNorm(32, out_scale*ch))
        self.net.append(Swish())
        self.net.append(nn.Conv2d(out_scale*ch, output_channels, kernel_size=3, padding=1))

    def forward(self, z):
        h = z
        for layer in self.net:
            h = layer(h)
        return h