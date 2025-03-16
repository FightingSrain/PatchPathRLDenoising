import os
import numpy as np
import cv2


class MiniBatchLoader(object):

    def __init__(self, train_path, train_path_gt, test_path, val_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.training_path_infos_gt = self.read_paths(train_path_gt, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)
        self.val_path_infos = self.read_paths(val_path, image_dir_path)

        self.crop_size = crop_size

        self.val_corp_size = 64
    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, self.training_path_infos_gt, indices, augment=True, color=True)

    def load_val_data(self, indices):
        return self.load_data(self.val_path_infos, self.training_path_infos_gt, indices, validation=True, color=True)

    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, self.training_path_infos_gt, indices, color=False)

    # test ok
    def load_data(self, path_infos, path_infos_gt, indices, augment=False, validation=False, color=False):
        mini_batch_size = len(indices)
        if color:
            in_channels = 3
        else:
            in_channels = 1

        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            target = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            for i, index in enumerate(indices):
                path = path_infos[index]
                path_gt = path_infos_gt[index]
                if color:
                    img = cv2.imread(path, 1)
                    img_gt = cv2.imread(path_gt, 1)
                else:
                    img = cv2.imread(path, 0)
                    img_gt = cv2.imread(path_gt, 0)

                h, w, c = img.shape
                if h < self.crop_size or w < self.crop_size:
                    continue
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    img_gt = np.fliplr(img_gt)

                if np.random.rand() > 0.5:
                    angle = 10 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))
                    img_gt = cv2.warpAffine(img_gt, M, (w, h))

                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size

                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                img = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                img_gt = img_gt[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                # print(img_gt[2])
                xs[i, :, :, :] = (img / 255.).transpose(2, 0, 1).astype(np.float32)
                target[i, :, :, :] = (img_gt / 255.).transpose(2, 0, 1).astype(np.float32)
                # print(target[i, :, :, :])
                # if color:
                #     xs[i, :, :, :] = (img / 255.).astype(np.float32)
                # else:
                #     xs[i, :, :, :] = (img / 255.).astype(np.float32)
            return xs, target

        elif validation:
            xs = np.zeros((mini_batch_size, in_channels, self.val_corp_size, self.val_corp_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path = path_infos[index]

                if color:
                    img = cv2.imread(path, 1)
                else:
                    img = cv2.imread(path, 0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))
                # xs[i, :, :, :] = cv2.resize(np.asarray(img),
                #                     (self.crop_size, self.crop_size),
                #                     interpolation=cv2.INTER_AREA).transpose(2,0,1) / 255.
                xs[i, :, :, :] = cv2.resize(np.asarray(img),
                                            (self.val_corp_size, self.val_corp_size),
                                            interpolation=cv2.INTER_AREA).transpose(2, 0, 1) / 255.

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path = path_infos[index]

                img = cv2.imread(path, 0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            h, w = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs[0, 0, :, :] = (img / 255.).astype(np.float32)

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs
