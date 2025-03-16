import numpy as np
import sys
import cv2
import copy

import torch
import torch.nn as nn
from utils import *
# from Net.DNN import DNN
from Net.DnCNN import DNN
import torch.optim as optim
from config import config
from utils import init_net, savevaltocsv, patin_val
from matplotlib import pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class State():
    def __init__(self, size, move_range):
        self.image = np.zeros(size, dtype=np.float32)
        self.label = np.zeros(size, dtype=np.float32)
        self.move_range = move_range
        self.dnn = init_net(DNN().to(device), 'kaiming', gpu_ids=[])
        self.dnn.load_state_dict(torch.load("./DNNSaveModel/44200_0.0009.pth"))
        self.opt_dnn = optim.Adam(self.dnn.parameters(), lr=config.LR)
        self.mse = nn.MSELoss()
        self.up_sample = nn.Upsample(scale_factor=4, mode='nearest')
        self.loss_all = []

    def reset(self, x, l):
        self.label = copy.deepcopy(l)
        self.image = copy.deepcopy(x)
        self.pre_img = copy.deepcopy(self.image)
        self.tensor = self.image

    def train_dnn(self, x, label, act, n_epi):
        B, H, W = act.shape
        act = np.random.randint(0, config.N_ACTIONS, size=(B, H, W))
        act = torch.Tensor(act.reshape(B, 1, H, W))

        act = self.up_sample(act)

        self.dnn.train()
        acts = torch.cat([act, act, act], 1).view(B*3, 1, act.shape[2], act.shape[3])
        res = self.dnn(torch.FloatTensor(x).cuda())[-1]
        acts = acts.type(torch.int64)
        res = res.gather(1, acts.cuda()).view(B, 3, act.shape[2], act.shape[3])

        loss = self.mse(res, torch.Tensor(label).cuda()) * 1
        if n_epi > 100:
            self.loss_all.append(loss.data.cpu().numpy())
        if n_epi % 100 == 0:
            print("+++++++++++++++++++++++++++")
            torch.save(self.dnn.state_dict(),
                       "./DNNSaveModel/{}_{:.4f}.pth".
                       format(n_epi, loss.data.cpu().numpy()))
            print("+++++++++++++++++++++++++++")
        print("\nLOSS: ", loss.data.cpu().numpy())
        print("+++++++++++++++++++++++++++")
        if n_epi % 100 == 0:
            plt.plot(self.loss_all)
            plt.pause(1)
            plt.close()

        self.opt_dnn.zero_grad()
        loss.backward()
        self.opt_dnn.step()


    def step(self, act, n_epi):
        B, H, W = act.shape

        layer1 = np.zeros(self.image.shape, self.image.dtype)
        layer2 = np.zeros(self.image.shape, self.image.dtype)
        layer3 = np.zeros(self.image.shape, self.image.dtype)
        layer4 = np.zeros(self.image.shape, self.image.dtype)
        layer5 = np.zeros(self.image.shape, self.image.dtype)


        # self.train_dnn(self.image, self.label, act, n_epi)

        self.dnn.eval()
        # l1, l2, l3, l4, _ = self.dnn(torch.Tensor(self.image).cuda())
        with torch.no_grad():
            act_t = self.up_sample(torch.Tensor(act.reshape(B, 1, H, W))).type(torch.int64).numpy()

            b, c, h, w = self.image.shape

            layers = self.dnn(torch.Tensor(self.image).cuda())
            for i in range(0, b):
                if np.sum(act[i] == 0) > 0:
                    # mask1 = np.where(act_t[i:i + 1] == 0, 1, 0)
                    # kernel = np.ones((5, 5), np.uint8)
                    # mask1_img = cv2.dilate(mask1[0][0]*1.0, kernel, iterations=1).astype(np.uint8)*self.image[i:i+1]
                    # layer1[i:i+1] = np.where(act_t[i:i + 1] == 0, mask1_img, 0)
                    mask1_img = self.image[i:i + 1]
                    # 切成多个4*4大小得patch
                    # patch = np.zeros((4, 4, 3))
                    # for j in range(0, 4):
                    #     for k in range(0, 4):
                    #         patch[j][k] = mask1_img[0][j*4:(j+1)*4, k*4:(k+1)*4]

                    # layer1[i:i+1] = self.dnn(torch.Tensor(mask1_img).cuda())[0].view(1, c, h, w).cpu().detach().numpy()

                    layer1[i:i + 1] = layers[0].view(b, c, h, w)[i:i + 1].cpu().detach().numpy()
                    self.image[i:i+1] = np.where(act_t[i:i+1] == 0, layer1[i:i+1], self.image[i:i+1])

                if np.sum(act[i] == 1) > 0:
                    # mask2 = np.where(act_t[i:i + 1] == 1, 1, 0)
                    # kernel = np.ones((5, 5), np.uint8)
                    # mask2_img = cv2.dilate(mask2[0][0]*1.0, kernel, iterations=1).astype(np.uint8)*self.image[i:i+1]
                    # layer2[i:i + 1] = np.where(act_t[i:i + 1] == 1, mask2_img, 0)

                    mask2_img = self.image[i:i + 1]
                    # layer2[i:i + 1] = self.dnn(torch.Tensor(mask2_img).cuda())[1].view(1, c, h, w).cpu().detach().numpy()
                    layer2[i:i + 1] = layers[1].view(b, c, h, w)[i:i + 1].cpu().detach().numpy()
                    self.image[i:i + 1] = np.where(act_t[i:i + 1] == 1, layer2[i:i + 1], self.image[i:i + 1])

                if np.sum(act[i] == 2) > 0:
                    # mask3 = np.where(act_t[i:i+1] == 2, 1, 0)
                    # kernel = np.ones((5, 5), np.uint8)
                    # mask3_img = cv2.dilate(mask3[0][0]*1.0, kernel, iterations=1).astype(np.uint8)*self.image[i:i+1]
                    # layer3[i:i + 1] = np.where(act_t[i:i+1] == 2, mask3_img, 0)

                    mask3_img = self.image[i:i + 1]
                    # layer3[i:i + 1] = self.dnn(torch.Tensor(mask3_img).cuda())[2].view(1, c, h, w).cpu().detach().numpy()
                    layer3[i:i + 1] = layers[2].view(b, c, h, w)[i:i + 1].cpu().detach().numpy()
                    self.image[i:i + 1] = np.where(act_t[i:i + 1] == 2, layer3[i:i + 1], self.image[i:i + 1])

                # if np.sum(act[i] == 3) > 0:
                #     # mask4 = np.where(act_t[i:i+1] == 3, 1, 0)
                #     # kernel = np.ones((5, 5), np.uint8)
                #     # mask4_img = cv2.dilate(mask4[0][0]*1.0, kernel, iterations=1).astype(np.uint8)*self.image[i:i+1]
                #     # layer4[i:i + 1] = np.where(act_t[i:i+1] == 3, mask4_img, 0)
                #
                #     mask4_img = self.image[i:i + 1]
                #     # layer4[i:i + 1] = self.dnn(torch.Tensor(mask4_img).cuda())[3].view(1, c, h, w).cpu().detach().numpy()
                #     layer4[i:i + 1] = layers[3].view(b, c, h, w)[i:i + 1].cpu().detach().numpy()
                #     self.image[i:i + 1] = np.where(act_t[i:i + 1] == 3, layer4[i:i + 1], self.image[i:i + 1])
                # if np.sum(act[i] == 4) > 0:
                #     # mask4 = np.where(act_t[i:i+1] == 3, 1, 0)
                #     # kernel = np.ones((5, 5), np.uint8)
                #     # mask4_img = cv2.dilate(mask4[0][0]*1.0, kernel, iterations=1).astype(np.uint8)*self.image[i:i+1]
                #     # layer4[i:i + 1] = np.where(act_t[i:i+1] == 3, mask4_img, 0)
                #
                #     mask4_img = self.image[i:i + 1]
                #     # layer4[i:i + 1] = self.dnn(torch.Tensor(mask4_img).cuda())[3].view(1, c, h, w).cpu().detach().numpy()
                #     layer5[i:i + 1] = layers[4].view(b, c, h, w)[i:i + 1].cpu().detach().numpy()
                #     self.image[i:i + 1] = np.where(act_t[i:i + 1] == 4, layer4[i:i + 1], self.image[i:i + 1])


        self.image = np.clip(self.image, a_min=0., a_max=1.)
        self.tensor[:, :self.image.shape[1], :, :] = self.image


