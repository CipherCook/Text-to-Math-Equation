import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, normalization=None, downsample=False, in_dim=32*32):
        super().__init__()
        
        self.downsample = downsample
        
        self.relu = nn.ReLU()
        
        if self.downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                normalization(out_channels) if normalization is not None else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                normalization(out_channels) if normalization is not None else nn.Identity()
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2),
                normalization(out_channels) if normalization is not None else nn.Identity(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                normalization(out_channels) if normalization is not None else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                normalization(out_channels) if normalization is not None else nn.Identity()
            )
    
    def forward(self, x):
        
        if self.downsample:
            return self.relu(self.downsampler(x) + self.block(x))

        return self.relu(x + self.block(x))


class ResNet(nn.Module):
    
    def __init__(self, n, r, normalization=None):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3,16,3),
            normalization(16) if normalization is not None else nn.Identity(),
            nn.ReLU(),
            nn.Sequential(*[ResNetBlock(16,16,normalization=normalization) for _ in range(n)]),
            ResNetBlock(16,32,downsample=True,in_dim=32*32,normalization=normalization),
            nn.Sequential(*[ResNetBlock(32,32,normalization=normalization) for _ in range(n-1)]),
            ResNetBlock(32,64,normalization=normalization,downsample=True,in_dim=16*16),
            nn.Sequential(*[ResNetBlock(64,64,normalization=normalization) for _ in range(n-1)]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,r)
        )
    
    def forward(self, x):
        return self.network(x)

def ResNet18(n, r):
    return ResNet(n, r)