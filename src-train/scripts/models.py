import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional 

class ESPCN(nn.Module):
    def __init__(self, img_channels, upscale_factor):
        super(ESPCN, self).__init__()
        self.img_channels = img_channels
        self.upscale_factor = upscale_factor

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(img_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, img_channels*(upscale_factor**2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        return x

class ESPCNSeq(nn.Module):
    def __init__(self, img_channels, upscale_factor):
        super(ESPCNSeq, self).__init__()
        self.img_channels = img_channels
        self.upscale_factor = upscale_factor
        self.body = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(img_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, img_channels*(upscale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor))


    def forward(self, x):
        x = self.body(x)
        return x


class FSRCNN(nn.Module):
    def __init__(self, img_channels, upscale_factor):
        """Fast Super-Resolution CNN.
        Inputs:
        - img_channels: either 1 (for Y-channel in YCbCr color space) or 3 (for RGB).
        - upscale_factor: upscale factor (2, 3, 4, 8).
        """
        super(FSRCNN, self).__init__()

        self.img_channels = img_channels
        self.upscale_factor = float(upscale_factor)

        self.extraction = nn.Sequential(
            nn.Conv2d(self.img_channels, 56, kernel_size=5, padding=2),
            nn.PReLU())
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, kernel_size=1),
            nn.PReLU())
        self.nonlinear_map = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(12, 12, kernel_size=3, padding=1),
            nn.PReLU(),
        )
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, kernel_size=1),
            nn.PReLU())
        self.deconv = nn.ConvTranspose2d(
            56, self.img_channels, kernel_size=9, padding=4, stride=upscale_factor,
            output_padding=upscale_factor-1)

    def forward(self, x):
        x = self.extraction(x)
        x = self.shrink(x)
        x = self.nonlinear_map(x)
        x = self.expand(x)
        x = self.deconv(x)

        return x

class SRCNN(nn.Module):
    def __init__(self, upscale_factor=2, img_channels=1):
        super(SRCNN, self).__init__()

        self.upscale_factor = float(upscale_factor)
        self.img_channels = img_channels

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.img_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, self.img_channels, kernel_size=5, padding=2)


    def forward(self, x):
        
        x = torch.nn.functional.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x