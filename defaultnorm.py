import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets

class ResNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, batch_norm_class=nn.BatchNorm2d, downsample=False, in_dim=32*32):
        super().__init__()
        
        self.downsample = downsample
        
        self.relu = nn.ReLU()
        
        if self.downsample:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                batch_norm_class(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                batch_norm_class(out_channels)
            )
            self.downsampler = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=2),
                batch_norm_class(out_channels),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
                batch_norm_class(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                batch_norm_class(out_channels)
            )
    
    def forward(self, x):
        
        if self.downsample:
            return self.relu(self.downsampler(x) + self.block(x))
        return self.relu(x + self.block(x))

class ResNet(nn.Module):
    
    def __init__(self, n, r, batch_norm_class=nn.BatchNorm2d):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Conv2d(3,16,3),
            batch_norm_class(16),
            nn.ReLU(),
            nn.Sequential(*[ResNetBlock(16,16,batch_norm_class=batch_norm_class) for _ in range(n)]),
            ResNetBlock(16,32,downsample=True,in_dim=32*32),
            nn.Sequential(*[ResNetBlock(32,32,batch_norm_class=batch_norm_class) for _ in range(n-1)]),
            ResNetBlock(32,64,batch_norm_class=batch_norm_class,downsample=True,in_dim=16*16),
            nn.Sequential(*[ResNetBlock(64,64,batch_norm_class=batch_norm_class) for _ in range(n-1)]),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(64,r)
        )
    
    def forward(self, x):
        return self.network(x)


# Create ResNet model instance
def ResNet18(n, r):
    return ResNet(n, r)