import torch
from torch import nn
import torch.nn.functional as F


class ResizeConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, scale_factor, mode="nearest"
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=1
        )

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class BasicBlockEnc(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, out_planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Enc(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1, 1], z_dim=128, nc=3):
        super().__init__()
        self.in_planes = 32
        self.conv1 = nn.Conv2d(nc, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.z_dim = z_dim
        self.layer1 = self._make_layer(BasicBlockEnc, 16, num_blocks[0], stride=2)
        self.layer2 = self._make_layer(BasicBlockEnc, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 128, num_blocks[3], stride=2)
        self.maxpool = nn.MaxPool2d(
            2, 2
        )  # changed from 4,4 to 2,2 to go from input size 512*512 to 256*256
        self.linear = nn.Linear(128 * 4 * 4, 2 * z_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, : self.z_dim]
        logvar = x[:, self.z_dim :]
        return mu, logvar


class BasicBlockDec(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv2 = nn.Conv2d(
            in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(in_planes)
        self.conv1 = ResizeConv2d(
            in_planes, out_planes, kernel_size=3, scale_factor=stride
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, out_planes, kernel_size=1, scale_factor=stride),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        # Adjust shortcut connection to match spatial dimensions
        shortcut_x = self.shortcut(x)
        if shortcut_x.size(2) != out.size(2) or shortcut_x.size(3) != out.size(3):
            shortcut_x = F.interpolate(
                shortcut_x, size=(out.size(2), out.size(3)), mode="nearest"
            )
        out += shortcut_x
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):
    def __init__(self, num_blocks=[1, 1, 1, 1], z_dim=128, nc=3):
        super().__init__()
        self.in_planes = z_dim
        self.linear = nn.Linear(z_dim, 128 * 16 * 16)
        self.layer4 = self._make_layer(BasicBlockDec, 128, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 64, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 32, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 16, num_blocks[0], stride=2)
        self.conv1 = ResizeConv2d(16, nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 128, 16, 16)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        return x


class ResVAE(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.encoder = ResNet18Enc(z_dim=z_dim)
        self.decoder = ResNet18Dec(z_dim=z_dim)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        x = self.decoder(z)
        return x, mean, logvar

    def reparameterize(self, mean, logvar):
        if self.train:
            std = torch.exp(logvar / 2)
            epsilon = torch.randn_like(std)
            return epsilon * std + mean
        else:
            return mean

    def loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction="mean")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE, KLD
