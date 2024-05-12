import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))

    def forward(self, x):
        N, C, H, W = x.size()
        G = self.num_groups
        assert C % G == 0, 'Number of channels must be divisible by number of groups'
        x = x.view(N, G, -1, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias
    
import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, normalization=None, downsample=False, in_dim=32*32):
        super().__init__()
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        
        if self.downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                normalization(num_groups=4, num_channels=out_channels) if normalization else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                normalization(num_groups=4, num_channels=out_channels) if normalization else nn.Identity()
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2),
                normalization(num_groups=4, num_channels=out_channels) if normalization else nn.Identity(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                normalization(num_groups=4, num_channels=out_channels) if normalization else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                normalization(num_groups=4, num_channels=out_channels) if normalization else nn.Identity()
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
            normalization(num_groups=4, num_channels=16) if normalization else nn.Identity(),
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
def ResNet_GN(n, r):
    return ResNet(n, r, GroupNorm)