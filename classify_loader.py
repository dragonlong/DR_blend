import torch.utils.data as data
import pickle
import h5py
import numpy as np
import torch
import random
import os
from data import load_augment
random.seed(334)


class data_loader(data.Dataset):
    def __init__(self, inputPath, ifTrain, transform = False):
        self.input_path = inputPath
        self.transform = transform
        self.file_list_total = os.listdir(inputPath)
        if ifTrain:
            self.file_list = self.file_list_total[: int(0.7*len(self.file_list_total))]
        else:
            self.file_list = self.file_list_total[int(0.7*len(self.file_list_total)): ]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.input_path + '/' + self.file_list[idx]

        w= 224
        h= 224
        aug_params= {                               
            'zoom_range': (1 / 1.15, 1.15),
            'rotation_range': (0, 360),
            'shear_range': (0, 0),
            'translation_range': (-20, 20),
            'do_flip': True,
            'allow_stretch': True,
        }
        sigma= 0.25
        image = load_augment(fname, w, h, aug_params=aug_params, transform=None, sigma=sigma, color_vec=None)
        #print('after', image.shape)
        data = h5py.File(self.input_path + '/' + self.file_list[idx], 'r')
        #image = data['image'].value
        #print('before', image.shape)
        target = float(data['target'].value)
        #one_hot = np.zeros(5)
        #one_hot[target] = 1
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.from_numpy(np.array([target]))


class FeatureLoader(data.Dataset):
    def __init__(self, base_path, isTrain, transform=None):
        self.inputPath = base_path
        self.transform = transform
        self.file_list_total = os.listdir(self.inputPath)
        if isTrain:
            self.file_list = self.file_list_total[:int(0.8*len(self.file_list_total))]
        else:
            self.file_list = self.file_list_total[int(0.8*len(self.file_list_total)):]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.inputPath+'/'+self.file_list[idx]
        f = h5py.File(fname, "r")
        data_mean = f.get('mean').value
        data_std = f.get('std').value
        data_label = float(f.get('label').value)
        data_vect = np.concatenate([data_mean, data_std], axis=0)
        # one_hot = np.zeros(5)
        # one_hot[data_label] = 1
        target = torch.from_numpy(np.array([data_label]))
        if self.transform is not None:
            data_vect = torch.from_numpy(data_vect)

        return data_vect, target


