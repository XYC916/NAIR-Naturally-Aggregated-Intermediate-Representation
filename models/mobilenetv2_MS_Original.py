'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, flag):
        super(Block, self).__init__()
        self.stride = stride
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if (flag == 0 and stride == 1 and in_planes != out_planes) or (flag == 1):
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


class MobileNetV2_MS_Original(nn.Module):
    cfg_head = [(1,  16, 1, 1),
                (6,  24, 2, 2)]

    cfg_layer = [(6, 32, 1, 2)]

    cfg_tail = [(6,  32, 2, 1),
                (6,  64, 4, 2),
                (6,  96, 3, 1),
                (6, 160, 3, 2),
                (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_MS_Original, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # self.layers = self._make_layers(in_planes=32)
        self.layers_head = self._make_layers_head(in_planes=32)
        self.layers_feat = self._make_layers_feat_list(channel=24)
        self.layers_tail = self._make_layers_tail(in_planes=32)

        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers_head(self, in_planes):
        layers = []
        flag = 0
        for expansion, out_planes, num_blocks, stride in self.cfg_head:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, flag))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def _make_layers_feat(self, in_planes):
        layers = []
        flag = 1
        for expansion, out_planes, num_blocks, stride in self.cfg_layer:
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, flag))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def _make_layers_feat_list(self, channel):
        layers = []
        channel_list = []
        for i in range(channel):
            if i % 2 == 0:
                channel_list.append(i + 2)
        for j in range(len(channel_list)):
            layers.append(self._make_layers_feat(channel_list[j]))
        return nn.Sequential(*layers)

    def _make_layers_tail(self, in_planes):
        layers = []
        flag = 0
        for expansion, out_planes, num_blocks, stride in self.cfg_tail:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, flag))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, flag, data):        # check data
        if flag == 1:  # train
            out_list = []
            out = F.relu6(self.bn1(self.conv1(x)))
            feat = out
            for i in range(len(self.layers_head)):
                feat = self.layers_head[i](feat)

            # every second channel
            feat = feat.split(2, dim=1)
            # sort channels under index
            index = [7, 11, 2, 0, 3, 9, 1, 5, 10, 4, 6, 8]
            feat_sort = []
            for x in index:
                feat_sort.append(feat[x])
            # combine
            feat_list = []
            feat_list.append(feat_sort[0])
            temp = torch.cat([feat_sort[0], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[2], feat_sort[0], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[2], feat_sort[0], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[2], feat_sort[4], feat_sort[0], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[2], feat_sort[4], feat_sort[0], feat_sort[5], feat_sort[1]],
                             dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[6], feat_sort[2], feat_sort[4], feat_sort[0], feat_sort[5],
                             feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[6], feat_sort[2], feat_sort[4], feat_sort[7], feat_sort[0],
                             feat_sort[5],feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[6], feat_sort[2], feat_sort[4], feat_sort[7], feat_sort[0],
                             feat_sort[5], feat_sort[8], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[6], feat_sort[2], feat_sort[4], feat_sort[9], feat_sort[7],
                             feat_sort[0], feat_sort[5], feat_sort[8], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[6], feat_sort[2], feat_sort[4], feat_sort[9], feat_sort[7],
                             feat_sort[10], feat_sort[0], feat_sort[5], feat_sort[8], feat_sort[1]], dim=1)
            feat_list.append(temp)
            temp = torch.cat([feat_sort[3], feat_sort[6], feat_sort[2], feat_sort[4], feat_sort[9], feat_sort[7],
                             feat_sort[10], feat_sort[0], feat_sort[11], feat_sort[5], feat_sort[8], feat_sort[1]],
                             dim=1)
            feat_list.append(temp)

            feat1 = self.layers_feat[0](feat_list[0])
            for i in range((len(self.layers_tail))):
                feat1 = self.layers_tail[i](feat1)
            feat1_out = feat1
            feat1_out = F.relu6(self.bn2(self.conv2(feat1_out)))
            feat1_out = F.avg_pool2d(feat1_out, 2)
            feat1_out = feat1_out.view(feat1_out.size(0), -1)
            feat1_out = self.linear(feat1_out)
            out_list.append(feat1_out)

            feat2 = self.layers_feat[1](feat_list[1])
            for i in range((len(self.layers_tail))):
                feat2 = self.layers_tail[i](feat2)
            feat2_out = feat2
            feat2_out = F.relu6(self.bn2(self.conv2(feat2_out)))
            feat2_out = F.avg_pool2d(feat2_out, 2)
            feat2_out = feat2_out.view(feat2_out.size(0), -1)
            feat2_out = self.linear(feat2_out)
            out_list.append(feat2_out)

            feat3 = self.layers_feat[2](feat_list[2])
            for i in range((len(self.layers_tail))):  # check
                feat3 = self.layers_tail[i](feat3)
            feat3_out = feat3
            feat3_out = F.relu6(self.bn2(self.conv2(feat3_out)))
            feat3_out = F.avg_pool2d(feat3_out, 2)
            feat3_out = feat3_out.view(feat3_out.size(0), -1)
            feat3_out = self.linear(feat3_out)
            out_list.append(feat3_out)

            feat4 = self.layers_feat[3](feat_list[3])
            for i in range((len(self.layers_tail))):  # check
                feat4 = self.layers_tail[i](feat4)
            feat4_out = feat4
            feat4_out = F.relu6(self.bn2(self.conv2(feat4_out)))
            feat4_out = F.avg_pool2d(feat4_out, 2)
            feat4_out = feat4_out.view(feat4_out.size(0), -1)
            feat4_out = self.linear(feat4_out)
            out_list.append(feat4_out)

            feat5 = self.layers_feat[4](feat_list[4])
            for i in range((len(self.layers_tail))):  # check
                feat5 = self.layers_tail[i](feat5)
            feat5_out = feat5
            feat5_out = F.relu6(self.bn2(self.conv2(feat5_out)))
            feat5_out = F.avg_pool2d(feat5_out, 2)
            feat5_out = feat5_out.view(feat5_out.size(0), -1)
            feat5_out = self.linear(feat5_out)
            out_list.append(feat5_out)

            feat6 = self.layers_feat[5](feat_list[5])
            for i in range((len(self.layers_tail))):  # check
                feat6 = self.layers_tail[i](feat6)
            feat6_out = feat6
            feat6_out = F.relu6(self.bn2(self.conv2(feat6_out)))
            feat6_out = F.avg_pool2d(feat6_out, 2)
            feat6_out = feat6_out.view(feat6_out.size(0), -1)
            feat6_out = self.linear(feat6_out)
            out_list.append(feat6_out)

            feat7 = self.layers_feat[6](feat_list[6])
            for i in range((len(self.layers_tail))):  # check
                feat7 = self.layers_tail[i](feat7)
            feat7_out = feat7
            feat7_out = F.relu6(self.bn2(self.conv2(feat7_out)))
            feat7_out = F.avg_pool2d(feat7_out, 2)
            feat7_out = feat7_out.view(feat7_out.size(0), -1)
            feat7_out = self.linear(feat7_out)
            out_list.append(feat7_out)

            feat8 = self.layers_feat[7](feat_list[7])
            for i in range((len(self.layers_tail))):  # check
                feat8 = self.layers_tail[i](feat8)
            feat8_out = feat8
            feat8_out = F.relu6(self.bn2(self.conv2(feat8_out)))
            feat8_out = F.avg_pool2d(feat8_out, 2)
            feat8_out = feat8_out.view(feat8_out.size(0), -1)
            feat8_out = self.linear(feat8_out)
            out_list.append(feat8_out)

            feat9 = self.layers_feat[8](feat_list[8])
            for i in range((len(self.layers_tail))):  # check
                feat9 = self.layers_tail[i](feat9)
            feat9_out = feat9
            feat9_out = F.relu6(self.bn2(self.conv2(feat9_out)))
            feat9_out = F.avg_pool2d(feat9_out, 2)
            feat9_out = feat9_out.view(feat9_out.size(0), -1)
            feat9_out = self.linear(feat9_out)
            out_list.append(feat9_out)

            feat10 = self.layers_feat[9](feat_list[9])
            for i in range((len(self.layers_tail))):  # check
                feat10 = self.layers_tail[i](feat10)
            feat10_out = feat10
            feat10_out = F.relu6(self.bn2(self.conv2(feat10_out)))
            feat10_out = F.avg_pool2d(feat10_out, 2)
            feat10_out = feat10_out.view(feat10_out.size(0), -1)
            feat10_out = self.linear(feat10_out)
            out_list.append(feat10_out)

            feat11 = self.layers_feat[10](feat_list[10])
            for i in range((len(self.layers_tail))):  # check
                feat11 = self.layers_tail[i](feat11)
            feat11_out = feat11
            feat11_out = F.relu6(self.bn2(self.conv2(feat11_out)))
            feat11_out = F.avg_pool2d(feat11_out, 2)
            feat11_out = feat11_out.view(feat11_out.size(0), -1)
            feat11_out = self.linear(feat11_out)
            out_list.append(feat11_out)

            feat12 = self.layers_feat[11](feat_list[11])
            for i in range((len(self.layers_tail))):  # check
                feat12 = self.layers_tail[i](feat12)
            feat12_out = feat12
            feat12_out = F.relu6(self.bn2(self.conv2(feat12_out)))
            feat12_out = F.avg_pool2d(feat12_out, 2)
            feat12_out = feat12_out.view(feat12_out.size(0), -1)
            feat12_out = self.linear(feat12_out)
            out_list.append(feat12_out)

            return out_list
        elif flag == 2:     # get data
            out = F.relu6(self.bn1(self.conv1(x)))
            f = out
            for i in range(len(self.layers_head)):
                f = self.layers_head[i](f)
            feat = f
            return feat
        elif flag == 0:  # test
            out = data
            out = self.layers_feat[11](out)

            for j in range(len(self.layers_tail)):
                out = self.layers_tail[j](out)

            out = F.relu6(self.bn2(self.conv2(out)))
            out = F.avg_pool2d(out, 2)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


def test():
    net = MobileNetV2_MS_Original()
    print(net)
    # x = torch.randn(1, 3, 32, 32)
    # print(x)
    # y = net(x, 1)
    # print(len(y))

# test()
