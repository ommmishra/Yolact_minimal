#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import logging
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, channel_reduce=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.cr = channel_reduce

        if self.cr:
            self.conv2 = nn.Conv2d(planes, planes // 4, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn2 = nn.BatchNorm2d(planes // 4, momentum=0.1)

            self.conv3 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes // 4, momentum=0.1)

        c_in = planes // 4 if self.cr else planes
        k = 1 if self.cr else 3
        self.conv4 = nn.Conv2d(c_in, planes, kernel_size=k, stride=1, padding=k // 2, bias=False)
        self.bn4 = nn.BatchNorm2d(planes, momentum=0.1)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.cr:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)
            out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False, padding=0)
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, *x):
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1, level_root=False, root_dim=0):
        super().__init__()
        channel_reduce = True if in_channels > 255 else False

        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride, channel_reduce)
            self.tree2 = block(out_channels, out_channels, 1, channel_reduce)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels, stride, root_dim=0)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels, root_dim=root_dim + out_channels)
        if levels == 1:
            self.root = Root(root_dim, out_channels)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels

        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                                         nn.BatchNorm2d(out_channels, momentum=0.1))

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)

        return x


class DLA34(nn.Module):
    def __init__(self, levels=(1, 1, 1, 3, 3, 2, 1), channels=(16, 32, 64, 128, 256, 512, 1024), block=BasicBlock):
        super().__init__()
        self.channels = channels
        self.base_layer = nn.Sequential(nn.Conv2d(3, channels[0], kernel_size=7, stride=1, padding=3, bias=False),
                                        nn.BatchNorm2d(channels[0], momentum=0.1),
                                        nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2, level_root=False)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2, level_root=True)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2, level_root=True)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2, level_root=True)
        self.level6 = Tree(levels[6], block, channels[5], channels[6], 2, level_root=True)

    @staticmethod
    def _make_conv_level(inplanes, planes, convs, stride=1):
        modules = []
        for i in range(convs):
            modules.extend(
                [nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride if i == 0 else 1, padding=1, bias=False),
                 nn.BatchNorm2d(planes, momentum=0.1),
                 nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        root_out = []
        x = self.base_layer(x)
        for i in range(len(self.channels)):
            x = getattr(self, f'level{i}')(x)
            root_out.append(x)

        selected_out = root_out[3:7]

        return selected_out

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = [128, 256, 512, 1024]
        self.downsample = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList([nn.Conv2d(self.in_channels[i], 256, kernel_size=3, padding=1) for i in range(4)])

    def forward(self, selected_out):
        fpn_outs = []
        for i in range(len(selected_out)):
            fpn_outs.append(F.relu(self.layers[i](selected_out[i])))

        fpn_outs.append(self.downsample(fpn_outs[-1]))

        return fpn_outs


class dla_yolact(nn.Module):
    def __init__(self):
        super().__init__()
        self.dla34 = DLA34()
        self.fpn = FPN()

    def forward(self, x):
        x = self.dla34(x)
        x = self.fpn(x)

        return x

    def init_backbone(self):
        state_dict = torch.load('weights/yolact_dla_init.pth')
        self.load_state_dict(state_dict, strict=False)

# net = dla_yolact()
#
# state_dict = net.state_dict()
# for k, v in state_dict.items():
#     print(k, list(v.shape))


# params = list(net.parameters())
#
# k = 0
# for i in params:
#     l = 1
#     for j in i.size():
#         l *= j
#     print(f'该层参数和：{str(l)}, shape：{str(list(i.size()))}')
#     k = k + l
# print("总参数数量和：" + str(k))

# input = torch.rand([1, 3, 512, 512])
# aa = net(input)


