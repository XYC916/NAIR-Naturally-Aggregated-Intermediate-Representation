'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
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
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2_IM(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2_IM, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x, flag):
        if flag == 1:   # train and test
            out = F.relu(self.bn1(self.conv1(x)))
            f = out
            for i in range(len(self.layers)):
                f = self.layers[i](f)
                # print(f.shape)
            out = f
            # out = self.layers(out)
            out = F.relu(self.bn2(self.conv2(out)))
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out
        if flag == 2:
            out = F.relu(self.bn1(self.conv1(x)))
            f = out
            for i in range(3):
                f = self.layers[i](f)
            out = f
            # for i in range(3):
            #     out = self.layers[i](out)
            return out
        if flag == 0:
            # out = data
            for i in range(3, len(self.layers)):
                out = self.layers[i](out)
            out = F.relu(self.bn2(self.conv2(out)))
            # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out

def test():
    net = MobileNetV2_IM()
    print(net)
    # x = torch.randn(1,3,32,32)
    # flag = 1
    # fit = 0
    # y = net(x, flag, fit)
    # print(y.size())

def test_bn():
    # encoding:utf-8
    import torch
    import torch.nn as nn
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # num_features - num_features from an expected input of size:batch_size*num_features*height*width
    m = nn.BatchNorm2d(2, affine=True)
    a = torch.randn(1, 2, 2, 4)
    b = torch.zeros(1, 2, 2, 4)
    input = torch.cat((a, b), dim=2)
    weight_tensor = m.weight.data.cpu().numpy()
    for i in range(weight_tensor.shape[0]):
        weight_tensor[i] = 0
    m.weight.data = torch.from_numpy(weight_tensor)
    output = m(input)

    print(input)
    print(input.shape)
    print(m.weight)
    print(m.bias)
    print(output)
    print(output.size())

# test()
# test_bn()
