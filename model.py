import torch
import torch.nn as nn


# input: 100 * 1 * 1 -> output: 3 * 64 * 64
class Generator(nn.Module):
    def __init__(self, z_dim, g_dim, channel):
        super(Generator, self).__init__()

        self.tconv1 = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(g_dim*8),
            nn.ELU(inplace=True)
        )

        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(g_dim * 8, g_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_dim*4),
            nn.ELU(inplace=True)
        )

        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(g_dim * 4, g_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_dim*2),
            nn.ELU(inplace=True)
        )

        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(g_dim * 2, g_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(g_dim),
            nn.ELU(inplace=True)
        )

        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(g_dim, channel, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        # weight initialization
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.normal_(mean=0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(mean=1, std=0.02)
                module.bias.data.fill_(0)

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        return x


# input: 3 * 64 * 64 -> output: 1
class Discriminator(nn.Module):
    def __init__(self, d_dim, channel):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, d_dim, 4, 2, 1, bias=False),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(d_dim, d_dim*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim*2),
            nn.ELU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(d_dim*2, d_dim*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim*4),
            nn.ELU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(d_dim*4, d_dim*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d_dim*8),
            nn.ELU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(d_dim*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        # weight initialization
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(mean=0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.normal_(mean=1, std=0.02)
                module.bias.data.fill_(0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return x4, x5.view(-1, 1).squeeze(1)  # squeeze to get scalar value

