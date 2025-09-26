
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv 3x3 -> GroupNorm -> ReLU) x2"""
    def __init__(self, in_channels: int, out_channels: int, *,
                 padding_mode: str = "zeros", padding: int = 1) -> None:
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=padding, padding_mode=padding_mode, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3,
                      padding=padding, padding_mode=padding_mode, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_op(x)


class DownSample(nn.Module):
    """DoubleConv puis MaxPool 2x2."""
    def __init__(self, in_channels: int, out_channels: int, *,
                 padding_mode: str = "zeros", padding: int = 1) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels,
                               padding_mode=padding_mode, padding=padding)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        down = self.conv(x)
        p = self.pool(down)
        return down, p


class UpSample(nn.Module):
    """ConvTranspose2d (x2), concat skip, DoubleConv."""
    def __init__(self, in_channels: int, out_channels: int, *,
                 padding_mode: str = "zeros", padding: int = 1) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2,
                                     kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels,
                               padding_mode=padding_mode, padding=padding)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """4-level U-Net; configurable padding_mode and padding."""
    def __init__(self, in_channels: int, num_classes: int, *,
                 padding_mode: str = "zeros", padding: int = 1) -> None:
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64,  padding_mode=padding_mode, padding=padding)
        self.down_convolution_2 = DownSample(64,         128, padding_mode=padding_mode, padding=padding)
        self.down_convolution_3 = DownSample(128,        256, padding_mode=padding_mode, padding=padding)
        self.down_convolution_4 = DownSample(256,        512, padding_mode=padding_mode, padding=padding)

        self.bottle_neck = DoubleConv(512, 1024, padding_mode=padding_mode, padding=padding)

        self.up_convolution_1 = UpSample(1024, 512, padding_mode=padding_mode, padding=padding)
        self.up_convolution_2 = UpSample(512,  256, padding_mode=padding_mode, padding=padding)
        self.up_convolution_3 = UpSample(256,  128, padding_mode=padding_mode, padding=padding)
        self.up_convolution_4 = UpSample(128,   64, padding_mode=padding_mode, padding=padding)

        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, p1 = self.down_convolution_1(x)
        d2, p2 = self.down_convolution_2(p1)
        d3, p3 = self.down_convolution_3(p2)
        d4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        u1 = self.up_convolution_1(b, d4)
        u2 = self.up_convolution_2(u1, d3)
        u3 = self.up_convolution_3(u2, d2)
        u4 = self.up_convolution_4(u3, d1)

        return self.out(u4)


class AutoRegUNet(nn.Module):
    """
    Autoregressive: generates n outputs concatenated along the channel dimension (C).
    """
    def __init__(self, in_channels: int, num_classes: int, n: int, *,
                 padding_mode: str = "zeros", padding: int = 1) -> None:
        super().__init__()
        if n < 1:
            raise ValueError("n must be >= 1")
        assert in_channels == num_classes, "AutoRegUNet requires in_channels == num_classes"
        self.unet = UNet(in_channels, num_classes, padding_mode=padding_mode, padding=padding)
        self.nstep = int(n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.unet(x)
        seq = out
        for _ in range(self.nstep - 1):
            out = self.unet(out)
            seq = torch.cat((seq, out), dim=1)
        return seq
