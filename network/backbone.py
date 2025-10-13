import math
import os
from typing import List

from torch.nn import functional as F

bn_mom = 0.0003

import torch
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        # self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))  # self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.shared_MLP(self.max_pool(x))  # self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, dilations=[1, 1, 1], stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False, padding=0, dilation=dilations[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilations[-2],
                               dilation=dilations[-2], bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False, padding=0, dilation=dilations[-1])
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args):
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        super(ResNet, self).__init__()
        if args.output_stride == 8:
            stride = [2, 1, 1]
            rates = [2, 4, 8]
        elif args.output_stride == 16:
            stride = [2, 2, 1]
            rates = [0, 2, 4]
        else:
            stride = [2, 2, 2]
            rates = [0, 0, 0]

        mult_grid = [1, 2, 4]

        self.downsample0 = nn.Conv2d(args.channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.downsample1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.downsample2 = self._make_layer(64, blocks=layers[0])
        self.downsample3 = self._make_layer(128, layers[1], stride=stride[0], rate=rates[0], mult_grid=mult_grid)
        self.downsample4 = self._make_layer(256, layers[2], stride=stride[1], rate=rates[1], mult_grid=mult_grid)
        self.downsample5 = self._make_layer(512, layers[3], stride=stride[2], rate=rates[2], mult_grid=mult_grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1, rate=0, mult_grid=[]):
        if rate == 0:
            dilations = []
        else:
            dilations = [mult_grid[i] * rate for i in range(len(mult_grid))]

        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * Bottleneck.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )

        layers = []
        if dilations != []:
            layers.append(Bottleneck(self.inplanes, planes, dilations, stride, downsample))
        else:
            layers.append(Bottleneck(self.inplanes, planes, [1, 1, 1], stride, downsample))
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            if dilations != []:
                layers.append(Bottleneck(self.inplanes, planes, dilations))
            else:
                layers.append(Bottleneck(self.inplanes, planes, [1, 1, 1]))
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.downsample0(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        x = self.downsample5(x)
        return x


class VGG16(nn.Module):
    def __init__(self, args):
        super(VGG16, self).__init__()
        self.conv11 = nn.Conv2d(in_channels=args.channel, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.concat1 = nn.ReLU(inplace=True)
        self.downsample1 = nn.MaxPool2d(2, stride=2)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.concat2 = nn.ReLU(inplace=True)
        self.downsample2 = nn.MaxPool2d(2, stride=2)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv34 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.concat3 = nn.ReLU(inplace=True)
        self.downsample3 = nn.MaxPool2d(2, stride=2)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv44 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.concat4 = nn.ReLU(inplace=True)
        self.downsample4 = nn.MaxPool2d(2, stride=2)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.downsample5 = nn.MaxPool2d(2, stride=2)

        self.relu = nn.ReLU(inplace=True)
        # self.fc1 = nn.Linear(512 * high//32 * width//32, 1024)
        # self.fc2 = nn.Linear(1024, number_classes)

    def forward(self, x):
        x = self.conv11(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv12(x)
        x = self.concat1(x)
        x = self.downsample1(x)

        x = self.conv21(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv22(x)
        x = self.concat2(x)
        x = self.downsample2(x)

        x = self.conv31(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv32(x)
        x = nn.ReLU(inplace=True)(x)

        x = self.conv34(x)
        x = self.concat3(x)
        x = self.downsample3(x)

        x = self.conv41(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv42(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv44(x)
        x = self.concat4(x)
        x = self.downsample4(x)

        # x = x.view(x.size(0), -1)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)

        return x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP_v3(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False,
                 activate_first=True, inplace=True):
        super(SeparableConv2d, self).__init__()
        self.relu0 = nn.ReLU(inplace=inplace)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                   groups=in_channels, bias=bias)
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_mom)
        self.relu1 = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_mom)
        self.relu2 = nn.ReLU(inplace=True)
        self.activate_first = activate_first

    def forward(self, x):
        if self.activate_first:
            x = self.relu0(x)
        x = self.depthwise(x)
        x = self.bn1(x)
        if not self.activate_first:
            x = self.relu1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        if not self.activate_first:
            x = self.relu2(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1, atrous=None, grow_first=True, activate_first=True,
                 inplace=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous_list = [atrous] * 3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters, momentum=bn_mom)
            self.head_relu = False
        else:
            self.skip = None

        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = SeparableConv2d(in_filters, filters, 3, stride=1, padding=1 * atrous[0],
                                        dilation=atrous[0], bias=False, activate_first=activate_first,
                                        inplace=self.head_relu)
        self.sepconv2 = SeparableConv2d(filters, out_filters, 3, stride=1, padding=1 * atrous[1],
                                        dilation=atrous[1], bias=False, activate_first=activate_first)
        self.sepconv3 = SeparableConv2d(out_filters, out_filters, 3, stride=strides, padding=1 * atrous[2],
                                        dilation=atrous[2], bias=False, activate_first=activate_first,
                                        inplace=inplace)

    def forward(self, inp):

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, args):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        stride_list = None
        if args.output_stride == 8:
            stride_list = [2, 1, 1]
        elif args.output_stride == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.' % os)
        self.downsample0 = nn.Conv2d(args.channel, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_mom)
        # do relu here

        self.downsample1 = Block(64, 128, 2)
        self.downsample2 = Block(128, 256, stride_list[0], inplace=False)
        self.downsample3 = Block(256, 728, stride_list[1])

        rate = 16 // args.output_stride
        self.middle_flow = Block(728, 728, 1, atrous=rate)

        self.downsample4 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.conv4 = SeparableConv2d(1536, 1536, 3, 1, 1 * rate, dilation=rate, activate_first=False)

        self.downsample5 = SeparableConv2d(1536, 2048, 3, 1, 1 * rate, dilation=rate, activate_first=False)

    def forward(self, input):
        self.layers = []
        x = self.downsample0(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        for i in range(16):
            x = self.middle_flow(x)
        x = self.downsample4(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.downsample5(x)
        return x
