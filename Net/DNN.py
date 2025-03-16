import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.utils.spectral_norm as spectral_norm
from torch.autograd import Variable
from numpy.random import randn
import numpy as np
import matplotlib.pyplot as plt


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.  conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        # self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.out = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=(1, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(64)
        self.bn7 = nn.BatchNorm2d(64)
        # self.bn8 = nn.BatchNorm2d(64)

        # Initialize the weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    nn.init.constant(m.bias, 0)
            elif classname.find('bn') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x_in):
        B, C, H, W = x_in.size()
        x = self.conv(x_in)
        x = F.relu(x)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        # ----------------
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        x3 = self.conv3(x2) + x1
        x3 = self.bn3(x3)
        x3 = F.relu(x3)
        # ----------------
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = F.relu(x4)
        x5 = self.conv5(x4) + x3
        x5 = self.bn5(x5)
        x5 = F.relu(x5)
        # ----------------
        x6 = self.conv6(x5)
        x6 = self.bn6(x6)
        x6 = F.relu(x6)
        x7 = self.conv7(x6) + x5
        x7 = self.bn7(x7)
        x7 = F.relu(x7)
        # ----------------
        # x8 = self.conv8(x7)
        # x8 = self.bn8(x8)
        # x8 = F.relu(x8)
        # x9 = self.conv9(x8) + x7
        # x9 = self.bn9(x9)
        # x9 = F.relu(x9)

        r1 = (torch.tanh(self.out(x1)) + x_in).view(B * C, 1, H, W)
        r2 = (torch.tanh(self.out(x3)) + x_in).view(B * C, 1, H, W)
        r3 = (torch.tanh(self.out(x5)) + x_in).view(B * C, 1, H, W)
        r4 = (torch.tanh(self.out(x7)) + x_in).view(B * C, 1, H, W)
        res = torch.cat((r1, r2, r3, r4), dim=1)
        return [r1, r2, r3, r4, res]
