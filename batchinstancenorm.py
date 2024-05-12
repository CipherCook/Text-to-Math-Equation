import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

#coding it
#eps to avoid zero division

class BatchInstanceNorm(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)
        self.instance_norm = nn.InstanceNorm2d(num_features)
        self.rho = nn.Parameter(torch.tensor([0.5]))
        self.weight = nn.Parameter(torch.ones(num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_features, 1, 1))

    def forward(self, x):
        bnorm = self.batch_norm(x)
        inorm = self.instance_norm(x)
        x_bin = self.weight * (self.rho * bnorm + (1 - self.rho) * inorm) + self.bias
        return x_bin

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=None, downsample=False):
        super(ResNetBlock, self).__init__()
        self.downsample = downsample
        self.relu = nn.ReLU()

        if self.downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                normalization(out_channels) if normalization else BatchInstanceNormalization(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                normalization(out_channels) if normalization else BatchInstanceNormalization(out_channels)
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                normalization(out_channels) if normalization else BatchInstanceNormalization(out_channels)
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                normalization(out_channels) if normalization else BatchInstanceNormalization(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                normalization(out_channels) if normalization else BatchInstanceNormalization(out_channels)
            )

    def forward(self, x):
        if self.downsample:
            return self.relu(self.downsampler(x) + self.block(x))
        return self.relu(x + self.block(x))

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes, normalization=None):
        super(ResNet, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            normalization(16) if normalization else BatchInstanceNormalization(16),
            nn.ReLU(),
            nn.Sequential(*[ResNetBlock(16, 16, normalization=normalization) for _ in range(num_blocks)]),
            ResNetBlock(16, 32, downsample=True, normalization=normalization),
            nn.Sequential(*[ResNetBlock(32, 32, normalization=normalization) for _ in range(num_blocks - 1)]),
            ResNetBlock(32, 64, downsample=True, normalization=normalization),
            nn.Sequential(*[ResNetBlock(64, 64, normalization=normalization) for _ in range(num_blocks - 1)]),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)

def ResNet_BIN(n, r):
    return ResNet(n, r, BatchInstanceNorm)

