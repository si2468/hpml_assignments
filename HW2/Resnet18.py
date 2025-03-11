# this code was taken from https://www.geeksforgeeks.org/resnet18-from-scratch-using-pytorch/

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, disable_bn=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.Identity() if disable_bn else nn.BatchNorm2d(out_channels)  # Remove BN if disabled
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.Identity() if disable_bn else nn.BatchNorm2d(out_channels)  # Remove BN if disabled

        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            if disable_bn:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, disable_bn=False, compile_mode="none"):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.Identity() if disable_bn else nn.BatchNorm2d(64)  # Remove BN if disabled
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.Sequential(
            self._make_layer(BasicBlock, 64, 2, stride=1, disable_bn=disable_bn),
            self._make_layer(BasicBlock, 128, 2, stride=2, disable_bn=disable_bn),
            self._make_layer(BasicBlock, 256, 2, stride=2, disable_bn=disable_bn),
            self._make_layer(BasicBlock, 512, 2, stride=2, disable_bn=disable_bn),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        self.compile_mode = compile_mode

        if compile_mode != "none":
            self.forward = torch.compile(self.forward, backend="inductor", mode=self.compile_mode)

    def _make_layer(self, block, out_channels, num_blocks, stride, disable_bn):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, disable_bn=disable_bn))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layers(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1) 
        out = self.fc(out)
        return out
