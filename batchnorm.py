import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets



class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            x = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return x * self.weight + self.bias
    
class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, normalization=None, downsample=False, in_dim=32*32):
        super().__init__()
        
        self.downsample = downsample
        
        self.relu = nn.ReLU()
        
        if self.downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                normalization(out_channels) if normalization else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                normalization(out_channels) if normalization else nn.Identity()
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2),
                normalization(out_channels) if normalization else nn.Identity(),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                normalization(out_channels) if normalization else nn.Identity(),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                normalization(out_channels) if normalization else nn.Identity()
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
            normalization(16) if normalization else nn.Identity(),
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
    
def ResNet_BN(n, r):
    return ResNet(n, r, BatchNorm)


