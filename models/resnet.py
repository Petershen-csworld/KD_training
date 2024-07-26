from __future__ import absolute_import

'''Resnet for cifar dataset.
Ported form
https://github.com/facebook/fb.resnet.torch
and
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from torch.autograd import Variable
import torch


__all__ = ['resnet']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    # Expansion factor used to increase the number of channels
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        # First 3x3 convolution layer
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        # Second 3x3 convolution layer
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)  # Batch normalization
        self.downsample = downsample  # Downsample layer to match dimensions
        self.stride = stride

    def forward(self, x):
        residual = x  # Save the input tensor as residual

        out = self.conv1(x)  # First convolution
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv2(out)  # Second convolution
        out = self.bn2(out)  # Batch normalization

        if self.downsample is not None:
            residual = self.downsample(x)  # Apply downsampling if necessary

        out += residual  # Add residual connection
        preact = out
        out = F.relu(out)  # ReLU activation
        if self.is_last:
            return out, preact  # Return both out and pre-activation if is_last is True
        else:
            return out  # Otherwise, return only out


class Bottleneck(nn.Module):
    # Expansion factor used to increase the number of channels
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        # First 1x1 convolution layer
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)  # Batch normalization
        # Second 3x3 convolution layer
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)  # Batch normalization
        # Third 1x1 convolution layer
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        self.downsample = downsample  # Downsample layer to match dimensions
        self.stride = stride

    def forward(self, x):
        residual = x  # Save the input tensor as residual

        out = self.conv1(x)  # First convolution
        out = self.bn1(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv2(out)  # Second convolution
        out = self.bn2(out)  # Batch normalization
        out = self.relu(out)  # ReLU activation

        out = self.conv3(out)  # Third convolution
        out = self.bn3(out)  # Batch normalization

        if self.downsample is not None:
            residual = self.downsample(x)  # Apply downsampling if necessary

        out += residual  # Add residual connection
        preact = out
        out = F.relu(out)  # ReLU activation
        if self.is_last:
            return out, preact  # Return both out and pre-activation if is_last is True
        else:
            return out  # Otherwise, return only out


class ResNet(nn.Module):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10, img_size=32):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (
                depth - 2) % 6 == 0, 'When using basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (
                depth - 2) % 9 == 0, 'When using bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name should be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(
            3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])  # Batch normalization
        self.relu = nn.ReLU(inplace=True)  # ReLU activation
        # Creating layers for the network
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)  # Average pooling
        # Fully connected layer
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        # Initializing weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # Helper function to create a layer of blocks
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # Downsampling if needed
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # Add the first block with possible downsampling
        layers.append(block(self.inplanes, planes, stride,
                      downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        # Add the remaining blocks
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                          is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        # Returns the list of feature modules
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        # Returns the batch normalization layers before ReLU activations
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(self, x, is_feat=False, preact=False):
        # Forward pass of the network
        x = self.conv1(x)  # Initial convolution
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)  # ReLU activation
        f0 = x

        x, f1_pre = self.layer1(x)  # First layer
        f1 = x
        x, f2_pre = self.layer2(x)  # Second layer
        f2 = x
        x, f3_pre = self.layer3(x)  # Third layer
        f3 = x

        # Global average pooling
        x = F.avg_pool2d(x, x.size(3))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        f4 = x
        x = self.fc(x)  # Fully connected layer

        if is_feat:
            if preact:
                # Return features before ReLU activations
                return [f0, f1_pre, f2_pre, f3_pre, f4], x
            else:
                # Return features after ReLU activations
                return [f0, f1, f2, f3, f4], x
        else:
            return x  # Return the final output


def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], 'basicblock', **kwargs)


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 32, 32)
    net = resnet8x4(num_classes=20)
    feats, logit = net(x, is_feat=True, preact=True)

    for f in feats:
        print(f.shape, f.min().item())
    print(logit.shape)

    for m in net.get_bn_before_relu():
        if isinstance(m, nn.BatchNorm2d):
            print('pass')
        else:
            print('warning')
