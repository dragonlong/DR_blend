import torch.utils.data as data
import pickle
import h5py
import numpy as np
import torch
import random
import os
import cv2
from data import load_augment
random.seed(334)

def dark_preserve(img, kernel = np.ones((25, 25), 'uint8')):
    img_opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img_opening

def bright_preserve(img, kernel = np.ones((25, 25), 'uint8')):
    img_closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img_closing

class data_loader(data.Dataset):
    def ori(self, w0, wf, i):
        return (0.975 ** (i - 1)) * w0 + (1 - 0.975 ** (i - 1)) * wf

    def sample_ratio(self, i):
        return self.ori(1.36, 1, i), self.ori(14.4, 2, i), self.ori(6.64, 2, i), self.ori(40.2, 2, i), self.ori(49.6, 2, i)

    def sampling(self, data, num):
        put = int(num/ len(data))
        need = num - put*len(data)
        sampled_items = random.sample(data, need)
        res = put*data + sampled_items
        random.shuffle(res)
        return res
    def __init__(self, inputPath, ifTrain, epoch, transform = False):
        self.inputPath = inputPath
        data_0 = pickle.load(open('/dev/shm/IB/sample_512_0.pkl', 'rb'))
        data_1 = pickle.load(open('/dev/shm/IB/sample_512_1.pkl', 'rb'))
        data_2 = pickle.load(open('/dev/shm/IB/sample_512_2.pkl', 'rb'))
        data_3 = pickle.load(open('/dev/shm/IB/sample_512_3.pkl', 'rb'))
        data_4 = pickle.load(open('/dev/shm/IB/sample_512_4.pkl', 'rb'))
        r0, r1, r2, r3, r4 = self.sample_ratio(epoch)
        deli = sum([r0, r1, r2, r3, r4])
        num0 = int(26000 * r0 / deli)
        num1 = int(26000 * r1 / deli)
        num2 = int(26000 * r2 / deli)
        num3 = int(26000 * r3 / deli)
        num4 = int(26000 * r4 / deli)
        data_0_train = data_0[: int(0.7 * len(data_0))]
        data_0_test = data_0[int(0.7 * len(data_0)):]
        data_1_train = data_1[: int(0.7 * len(data_1))]
        data_1_test = data_1[int(0.7 * len(data_1)):]
        data_2_train = data_2[: int(0.7 * len(data_2))]
        data_2_test = data_2[int(0.7 * len(data_2)):]
        data_3_train = data_3[: int(0.7 * len(data_3))]
        data_3_test = data_3[int(0.7 * len(data_3)):]
        data_4_train = data_4[: int(0.7 * len(data_4))]
        data_4_test = data_4[int(0.7 * len(data_4)):]

        self.file_list_train = self.sampling(data_0_train, num0) + self.sampling(data_1_train, num1) + self.sampling(
            data_2_train, num2) + self.sampling(data_3_train, num3) + self.sampling(data_4_train, num4)
        random.shuffle(self.file_list_train)

        # self.input_path = inputPath
        self.transform = transform

        if ifTrain:
            self.file_list = self.file_list_train
        else:
            self.file_list = data_0_test + data_1_test + data_2_test + data_3_test + data_4_test

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        fpath = os.path.join(self.inputPath, self.file_list[idx].split('/')[-1])
        #fpath = self.file_list[idx]
        w= 512
        h= 512
        aug_params= {
            'zoom_range': (1 / 1.15, 1.15),
            'rotation_range': (0, 360),
            'shear_range': (0, 0),
            'translation_range': (-20, 20),
            'do_flip': True,
            'allow_stretch': True,
        }
        sigma= 0.25
        #image = load_augment(fname, w, h, aug_params=aug_params, transform=None, sigma=sigma, color_vec=None)

        #print('after', image.shape)
        data = h5py.File(fpath, 'r')
        image = data['image'].value
        #image_bright = bright_preserve(image)
        #image_dark = dark_preserve(image)
        #image_comb = np.concatenate((image, image_bright, image_dark), axis=2)
        #print('before', image.shape)
        target = float(data['target'].value)
        #target = int(data['target'].value)
        #one_hot = np.zeros(5)
        #one_hot[target] = 1
        if self.transform is not None:
            image_comb = self.transform(image)
        #return image, torch.from_numpy(np.array([target]))
        #return image, torch.from_numpy(one_hot)
        return image_comb, target