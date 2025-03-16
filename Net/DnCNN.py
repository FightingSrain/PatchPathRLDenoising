import torch
import torch.nn as nn

import Net.basicblock as B
from thop import profile


# DnCNN
class DNN(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nc=64, nb=17, act_mode='BR'):
        """
        # ------------------------------------
        in_nc: channel number of input
        out_nc: channel number of output
        nc: channel number
        nb: total number of conv layers
        act_mode: batch norm + activation function; 'BR' means BN+ReLU.
        # ------------------------------------
        Batch normalization and residual learning are
        beneficial to Gaussian denoising (especially
        for a single noise level).
        The residual of a noisy image corrupted by additive white
        Gaussian noise (AWGN) follows a constant
        Gaussian distribution which stablizes batch
        normalization during training.
        # ------------------------------------
        """
        super(DNN, self).__init__()
        assert 'R' in act_mode or 'L' in act_mode, 'Examples of activation function: R, L, BR, BL, IR, IL'
        bias = True

        self.m_head = B.conv(in_nc, nc, mode='C'+act_mode[-1], bias=bias)
        self.bolck1 = nn.Sequential(
            B.conv(nc, nc, mode='C'+act_mode, bias=bias),
            B.conv(nc, nc, mode='C'+act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
        )

        self.bolck2 = nn.Sequential(
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
        )

        self.bolck3 = nn.Sequential(
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
        )

        self.bolck4 = nn.Sequential(
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
        )

        self.bolck5 = nn.Sequential(
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
            B.conv(nc, nc, mode='C' + act_mode, bias=bias),
        )

        # m_body = [B.conv(nc, nc, mode='C'+act_mode, bias=bias) for _ in range(nb-2)]
        self.m_tail = B.conv(nc, out_nc, mode='C', bias=bias)

        # self.model = B.sequential(m_head, *m_body, m_tail)



    def forward_(self, x):
        B, C, H, W = x.size()

        x_head = self.m_head(x)
        x1 = self.bolck1(x_head)
        x2 = self.bolck2(x1)
        x3 = self.bolck3(x2)
        x4 = self.bolck4(x3)
        x5 = self.bolck5(x4)

        n1 = (x - self.m_tail(x1)).view(B * C, 1, H, W)
        n2 = (x - self.m_tail(x2)).view(B * C, 1, H, W)
        n3 = (x - self.m_tail(x3)).view(B * C, 1, H, W)
        n4 = (x - self.m_tail(x4)).view(B * C, 1, H, W)
        n5 = (x - self.m_tail(x5)).view(B * C, 1, H, W)

        res = torch.cat((n1, n2, n3, n4, n5), dim=1)
        return [n1, n2, n3, n4, n5, res]
        # res = torch.cat((n1, n3, n5), dim=1)
        # return [n1, n3, n5, res]

    def forward(self, x, l):
        B, C, H, W = x.size()

        x_head = self.m_head(x)

        if l == 1:
            x1 = self.bolck1(x_head)
            n1 = (x - self.m_tail(x1)).view(B * C, 1, H, W)
            return n1

        if l == 2:
            x1 = self.bolck1(x_head)
            x2 = self.bolck2(x1)
            n2 = (x - self.m_tail(x2)).view(B * C, 1, H, W)
            return n2

        if l == 3:
            x1 = self.bolck1(x_head)
            x2 = self.bolck2(x1)
            x3 = self.bolck3(x2)
            n3 = (x - self.m_tail(x3)).view(B * C, 1, H, W)
            return n3

        if l == 4:
            x1 = self.bolck1(x_head)
            x2 = self.bolck2(x1)
            x3 = self.bolck3(x2)
            x4 = self.bolck4(x3)
            n4 = (x - self.m_tail(x4)).view(B * C, 1, H, W)
            return n4
        # x5 = self.bolck5(x4)
        if l == 5:
            x1 = self.bolck1(x_head)
            x2 = self.bolck2(x1)
            x3 = self.bolck3(x2)
            x4 = self.bolck4(x3)
            x5 = self.bolck5(x4)
            n5 = (x - self.m_tail(x5)).view(B * C, 1, H, W)
            return n5

