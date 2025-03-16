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

class Actor(nn.Module):
    def __init__(self, n_act=4):
        super(Actor, self).__init__()
        self.data = []
        self.n_act = n_act
        nf = 64
        self.nf = nf
        self.patch_size = 4
        # self.conv = nn.Sequential(
        #     (nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)),
        #     nn.ReLU(),
        #     (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2, bias=True)),
        #     nn.ReLU(),
        #     (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3, bias=True)),
        #     nn.ReLU(),
        #     (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(4, 4), dilation=4, bias=True)),
        #     nn.ReLU(),
        # )
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2, bias=True)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(4, 4), dilation=4, bias=True)


        # self.max = nn.AdaptiveMaxPool2d((4, 4))

        self.maxp = nn.MaxPool2d(self.patch_size, stride=self.patch_size, padding=0)

        self.diconv1_p1 = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                                   bias=True))
        self.diconv2_p1 = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                                   bias=True))
        # b, 64, w, h
        # self.conv1d = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(1, 1), bias=True)

        self.out_pi = nn.Conv2d(in_channels=64, out_channels=self.n_act * 3, kernel_size=3, stride=1, padding=(1, 1),
                                 bias=True)
        # self.out_pi2 = nn.Conv2d(in_channels=64, out_channels=self.n_act, kernel_size=3, stride=1, padding=(1, 1),
        #                         bias=True)
        # self.out_pi3 = nn.Conv2d(in_channels=64, out_channels=self.n_act, kernel_size=3, stride=1, padding=(1, 1),
        #                          bias=True)

        self.diconv1_v = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(3, 3), dilation=3,
                                   bias=True))
        self.diconv2_v = (nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=(2, 2), dilation=2,
                                   bias=True))
        self.value = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=(1, 1), bias=True)

        # 上采样
        self.upsample = nn.Upsample(scale_factor=self.patch_size, mode='nearest')

        # kernel = torch.zeros((1, 1, 33, 33))
        # kernel[:, :, 16, 16] = 1
        # self.weight = nn.Parameter(data=kernel, requires_grad=True)
        # self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=False)

    # RMC
    def conv_smooth(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=16)
        return x

    def forward(self, x):
        B, C, H, W = x.size()
        # 如果输入的图片大小不是patch_size的整数倍，则进行填充
        if H % self.patch_size != 0 or W % self.patch_size != 0:
            x = F.pad(x, (0, self.patch_size - W % self.patch_size, 0, self.patch_size - H % self.patch_size), mode='replicate')

        conv1 = self.conv1(x)
        conv1 = F.relu(conv1)
        # conv1 = self.maxp(conv1)
        conv2 = self.conv2(conv1)
        conv2 = F.relu(conv2)
        conv3 = self.conv3(conv2)
        conv3 = F.relu(conv3)
        conv4 = self.conv4(conv3)
        conv4 = F.relu(conv4)

        # conv4 = self.maxp(conv4)

        p1 = self.diconv1_p1(conv4)
        p1 = F.relu(p1)
        p1 = self.diconv2_p1(p1)
        p1 = F.relu(p1)

        policy1 = F.softmax(self.out_pi(p1)[:, 0:self.n_act, :, :], 1)
        policy2 = F.softmax(self.out_pi(p1)[:, self.n_act:self.n_act * 2, :, :], 1)
        policy3 = F.softmax(self.out_pi(p1)[:, self.n_act * 2:self.n_act * 3, :, :], 1)

        policy = torch.cat((policy1, policy2, policy3), 1)

        policy = self.maxp(policy)

        v = self.diconv1_v(conv4)
        v = F.relu(v)
        v = self.diconv2_v(v)
        v = F.relu(v)
        value = self.value(v)
        value = self.maxp(value)

        return policy, value

# test
# if __name__ == '__main__':
#     model = Actor()
#     x = torch.randn((1, 3, 32, 32))
#     policy, value = model(x)
#     plt.imshow(np.argmax(policy[0].detach().numpy(), axis=1))
#     # plt.imshow(policy[0, 0].detach().numpy())
#     # plt.imshow(value[0, 0].detach().numpy())
#     plt.show()
#     print(policy.shape, value.shape)

