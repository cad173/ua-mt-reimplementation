import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    Double Convolution block: conv2d -> BN -> LeakyReLU -> [Dropout] -> conv2d -> BN -> LeakyReLU
    """

    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.0):
        super(DownSample, self).__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Decoder(nn.Module):
    """Decoder stage. SSL4MIS uses NO dropout in the decoder path (all noise /
    regularization lives in the encoder), so there is no dropout layer here."""
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        target_size = skip.shape[2:]
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size)

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Output Convolution block: 1x1 convolution to map to the required number of classes
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class UnetMCDoptout(nn.Module):
    """
    complete model
    """
    def __init__(self, n_channels, n_classes, base_ch=64):
        super(UnetMCDoptout, self).__init__()

        b = base_ch

        # SSL4MIS element wise dropout schedule
        drop = [0.05, 0.10, 0.20, 0.30, 0.50]

        # Input layer
        self.inc = DoubleConv(n_channels, b, drop[0])

        # Encoder
        self.down1 = DownSample(b, b * 2, drop[1])
        self.down2 = DownSample(b * 2, b * 4, drop[2])
        self.down3 = DownSample(b * 4, b * 8, drop[3])
        self.down4 = DownSample(b * 8, b * 16, drop[4])

        # Decoder
        self.up1 = Decoder(b * 16, b * 8)
        self.up2 = Decoder(b * 8, b * 4)
        self.up3 = Decoder(b * 4, b * 2)
        self.up4 = Decoder(b * 2, b)

        # Output Layer
        self.outc = OutConv(b, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.outc(x)
    

    