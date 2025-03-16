import torch.nn.functional as F
import numpy as np
from Net.unet_part import *


class Actor(nn.Module):
    def __init__(self,  n_act=9, bilinear=True):
        super(Actor, self).__init__()
        self.n_channels = 3
        self.n_act = n_act
        self.bilinear = bilinear


        nf = 32
        self.nf = nf
        self.inc = DoubleConv(3, nf)
        self.down1 = Down(nf, nf * 2)
        self.down2 = Down(nf * 2, nf * 4)
        self.down3 = Down(nf * 4, nf * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(nf * 8, nf * 16 // factor)

        self.up1_a = Up(nf * 16, nf * 8 // factor, bilinear)
        self.up2_a = Up(nf * 8, nf * 4 // factor, bilinear)
        self.up3_a = Up(nf * 4, nf * 2 // factor, bilinear)
        self.up4_a = Up(nf * 2, nf, bilinear)
        self.conv7_Wz1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Uz1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Wr1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Ur1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_W1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_U1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)

        self.outc_piRGB = OutConv1(nf, self.n_act*3)

        self.up1_c = Up(nf * 16, nf * 8 // factor, bilinear)
        self.up2_c = Up(nf * 8, nf * 4 // factor, bilinear)
        self.up3_c = Up(nf * 4, nf * 2 // factor, bilinear)
        self.up4_c = Up(nf * 2, nf, bilinear)
        self.conv7_Wz2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Uz2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Wr2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_Ur2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_W2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)
        self.conv7_U2 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=3, stride=1, padding=(1, 1), bias=False)

        kernel = torch.zeros((1, 1, 33, 33))
        kernel[:, :, 16, 16] = 1
        self.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(1), requires_grad=False)

        self.outc_meanRGB = OutConv1(nf, self.n_act * 3)
        self.outc_logstdRGB = nn.Parameter(torch.zeros(1, self.n_act * 3), requires_grad=True)
        #______

        self.up1 = Up(nf * 16, nf * 8 // factor, bilinear)
        self.up2 = Up(nf * 8, nf * 4 // factor, bilinear)
        self.up3 = Up(nf * 4, nf * 2 // factor, bilinear)
        self.up4 = Up(nf * 2, nf, bilinear)
        self.value = OutConv1(nf, 3)

    def conv_smooth(self, x):
        x = F.conv2d(x, self.weight, self.bias, stride=1, padding=16)
        return x

    def parse_p(self, u_out):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        return p

    def forward(self, x):
        B, C, H, W = x[:, 0:3, :, :].size()
        x_in = x[:, 0:3, :, :]
        ht1 = x[:, 3:self.nf + 3, :, :]
        ht2 = x[:, self.nf + 3:self.nf + 3 + self.nf, :, :]

        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        xa = self.up1_a(x5, x4)
        xa = self.up2_a(xa, x3)
        xa = self.up3_a(xa, x2)
        xa = self.up4_a(xa, x1)
        GRU_in1 = xa
        z_t = torch.sigmoid(self.conv7_Wz1(GRU_in1) + self.conv7_Uz1(ht1))
        r_t = torch.sigmoid(self.conv7_Wr1(GRU_in1) + self.conv7_Ur1(ht1))
        h_title_t = torch.tanh(self.conv7_W1(GRU_in1) + self.conv7_U1(r_t * ht1))
        h_t1 = (1 - z_t) * ht1 + z_t * h_title_t

        policyRGB = self.outc_piRGB(h_t1).reshape(B * C, self.n_act, H, W)
        policy = F.softmax(policyRGB, 1)

        xc = self.up1_c(x5, x4)
        xc = self.up2_c(xc, x3)
        xc = self.up3_c(xc, x2)
        xc = self.up4_c(xc, x1)
        GRU_in2 = xc
        z_t = torch.sigmoid(self.conv7_Wz2(GRU_in2) + self.conv7_Uz2(ht2))
        r_t = torch.sigmoid(self.conv7_Wr2(GRU_in2) + self.conv7_Ur2(ht2))
        h_title_t = torch.tanh(self.conv7_W2(GRU_in2) + self.conv7_U2(r_t * ht2))
        h_t2 = (1 - z_t) * ht2 + z_t * h_title_t

        meanRGB = self.parse_p(self.outc_meanRGB(h_t2)).reshape(B * 3, self.n_act)
        logstdRGB = self.outc_logstdRGB.expand([B, self.n_act * 3]).reshape(B * 3, self.n_act, 1, 1)


        xv = self.up1(x5, x4)
        xv = self.up2(xv, x3)
        xv = self.up3(xv, x2)
        xv = self.up4(xv, x1)
        value = self.value(xv).reshape(B*3, 1, H, W)

        return policy, value, meanRGB, logstdRGB, h_t1, h_t2
# actor = Actor(config)
# ins = torch.zeros((16, 3, 128, 128))
# p, v, m, std = actor(ins, 16, 2)