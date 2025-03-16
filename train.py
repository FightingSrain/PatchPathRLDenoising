import copy
import time
import math
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from config import config
import State as State
from pixelwise_a3c import PixelWiseA3C_InnerState
from utils import init_net, savevaltocsv, patin_val
from mini_batch_loader import MiniBatchLoader
# from Net.unet_gru import Actor
from Net.FCN_sample import Actor
from tqdm import tqdm
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# TRAINING_DATA_PATH = "lol_low.txt"
# TRAINING_DATA_PATH_GT = "lol_high.txt"
# TESTING_DATA_PATH = "lol_low.txt"

TRAINING_DATA_PATH = "train.txt"
TRAINING_DATA_PATH_GT = "train.txt"
TESTING_DATA_PATH = "train.txt"

VAL_DATA_PATH = "val.txt"
IMAGE_DIR_PATH = "..//"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

torch.manual_seed(1234)
np.random.seed(1234)

up_sample = nn.Upsample(scale_factor=4, mode='nearest')


def main():
    model = init_net(Actor(n_act=config.N_ACTIONS).to(device), 'orthogonal', gpu_ids=[])
    optimizer = optim.Adam(model.parameters(), lr=config.LR)
    i_index = 0

    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TRAINING_DATA_PATH_GT,
        TESTING_DATA_PATH,
        VAL_DATA_PATH,
        IMAGE_DIR_PATH,
        config.corp_size)

    current_state = State.State((config.BATCH_SIZE, 3, config.img_size, config.img_size), config.MOVE_RANGE)
    agent = PixelWiseA3C_InnerState(model, optimizer, config.BATCH_SIZE, config.EPISODE_LEN, config.GAMMA)

    # train dataset
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)

    # val dataset
    # val_data_size = MiniBatchLoader.count_paths(VAL_DATA_PATH)
    # indices_val = np.random.permutation(val_data_size)
    #
    # r_val = indices_val
    # raw_val = mini_batch_loader.load_val_data(r_val)
    # len_val = len(raw_val)
    # ValData = []
    # pre_pnsr = -math.inf

    # torch.cuda.empty_cache()
    for n_epi in tqdm(range(0, 100000), ncols=70, initial=0):

        r = indices[i_index: i_index + config.BATCH_SIZE]
        raw_x, labels = mini_batch_loader.load_training_data(r)

        # denoise
        raw_n = np.random.normal(0, config.sigma, labels.shape).astype(labels.dtype) / 255.
        ori_ins_noisy = np.clip(labels + raw_n, a_min=0., a_max=1.)

        # img enhancement
        # ori_ins_noisy = copy.deepcopy(raw_x)
        label = copy.deepcopy(labels)

        if n_epi % 10 == 0:
            # print(label[2])
            image = np.asanyarray(label[2].transpose(1, 2, 0) * 255, dtype=np.uint8)
            image = np.squeeze(image)
            cv2.imshow("label", image)
            cv2.waitKey(1)

        # if n_epi % 10 == 0:
        #     # print(label[2])
        #     image = np.asanyarray(ori_ins_noisy[2].transpose(1, 2, 0) * 255, dtype=np.uint8)
        #     image = np.squeeze(image)
        #     cv2.imshow("img", image)
        #     cv2.waitKey(1)
        current_state.reset(ori_ins_noisy.copy(), label.copy())
        reward = np.zeros((config.BATCH_SIZE, 1, config.img_size, config.img_size))
        sum_reward = 0

        for t in range(config.EPISODE_LEN):
            # previous_image = np.clip(copy.deepcopy(current_state.image), a_min=0., a_max=1.)

            action, action_prob, pout = agent.act_and_train(current_state.tensor, reward)

            # _, tst_act = torch.max(torch.Tensor(pout), dim=1)
            tst_act = torch.argmax(torch.Tensor(pout), dim=1)
            tst_act = tst_act.unsqueeze(1).type(torch.FloatTensor)

            # print(tst_act)
            if n_epi % 50 == 0:
                # print(model.upsample(torch.FloatTensor(tst_act)))
                print(tst_act[2])
                print(action_prob[2])
                paint_amap(model.upsample(tst_act).numpy().squeeze()[2], current_state.image[2])
                # paint_scatter(tst_act[10], current_state.image[10])
            current_state.step(action, n_epi)
            if n_epi % 10 == 0:
                image = np.asanyarray(current_state.image[2].transpose(1, 2, 0) * 255, dtype=np.uint8)
                image = np.squeeze(image)
                cv2.imshow("temp", image)
                cv2.waitKey(1)

            reward1 = (-np.abs(label - current_state.image))

            reward2 = np.zeros((config.BATCH_SIZE, 1, config.img_size, config.img_size))
            for i in range(config.N_ACTIONS - 1):
                for j in range(i + 1, config.N_ACTIONS):
                    reward2[:, 0:1] += (
                        np.abs(model.upsample(torch.Tensor(pout[:, i:i + 1, :, :] - pout[:, j:j + 1, :, :])).numpy()))
            reward2 = (config.N_ACTIONS - reward2)

            reward3 = np.zeros((config.BATCH_SIZE, 1, config.img_size, config.img_size))
            for i in range(config.N_ACTIONS):
                tmp = np.zeros((config.BATCH_SIZE, 1, config.img_size, config.img_size))
                for j in range(config.BATCH_SIZE):
                    tmp += model.upsample(torch.Tensor(pout[j:j + 1, i:i + 1])).numpy()
                reward3[:, 0:1] += -np.abs(tmp - config.BATCH_SIZE / config.N_ACTIONS)
            reward3 /= (config.BATCH_SIZE / config.N_ACTIONS) * 4
            # print(reward1[0])
            # print(reward2[0])
            # print(reward3[0])
            print("---------------------------------------------------------------------")
            reward = reward1 * 2000 + reward2 * 1 + reward3 * 6
            # reward = reward1*255

            # reward = 255 * (- np.square(label - current_state.image) - model.upsample(torch.Tensor(np.expand_dims(action, 1))).numpy())
            # print(reward.shape)
            # print("TTTR##$#$")
            # reward = reward.reshape(reward.shape[0], 1, reward.shape[2], reward.shape[3])
            sum_reward += np.mean(reward) * np.power(config.GAMMA, t)

        if n_epi > 1 and n_epi % 1 == 0:
            agent.stop_episode_and_train(current_state.tensor, reward, True)

        torch.cuda.empty_cache()

        # if n_epi % 100 == 0 and n_epi != 0:
        #     temp_psnr, temp_ssim = agent.val(agent, State, raw_val, config.EPISODE_LEN)
        #     if temp_psnr > pre_pnsr:
        #         pre_pnsr = temp_psnr
        #         for f in os.listdir("./HybridModel_{}/".format(config.sigma)):
        #             if os.path.splitext(f)[1] == ".pth":
        #                 os.remove("./HybridModel_{}/{}".format(config.sigma, f))
        #         torch.save(model.state_dict(),
        #                "./HybridModel_{}/{}_{}_{}.pth".
        #                    format(config.sigma, n_epi, temp_psnr, temp_ssim))
        #         print("save model")
        #     ValData.append([n_epi, temp_psnr, temp_ssim])
        #     savevaltocsv(ValData, "val.csv", config.sigma)  # 保存验证集数据
        #     patin_val(np.asarray(ValData)[:, 1])

        if i_index + config.BATCH_SIZE >= train_data_size:
            i_index = 0
            indices = np.random.permutation(train_data_size)
        else:
            i_index += config.BATCH_SIZE

        if i_index + 2 * config.BATCH_SIZE >= train_data_size:
            i_index = train_data_size - config.BATCH_SIZE

        print("train total reward {a}".format(a=sum_reward * 255))


def paint_amap(acmap, img):
    act_map = np.asanyarray(acmap.squeeze(), dtype=np.uint8)
    # act_map归一化到0-1
    # act_map = (act_map - np.min(act_map)) / (np.max(act_map) - np.min(act_map))
    img = np.asanyarray(img.squeeze() * 255, dtype=np.uint8).transpose(1, 2, 0)
    # plt.imshow(img)
    # plt.imshow(act_map, alpha=0.4)
    plt.imshow(act_map, vmax=config.N_ACTIONS - 1, vmin=0)
    plt.colorbar()
    plt.pause(1.5)
    plt.close('all')
    # plt.show()


if __name__ == '__main__':
    main()
