import h5py
import numpy as np
import torch
import os
base_path = "/media/dragonx/DataStorage/download/5label_mean_std/"
fname = os.path.join(base_path, "16_left.hdf5")
f = h5py.File(fname, "r")
data_mean = f.get('mean').value
data_std = f.get('std').value
data_label = int(f.get('label').value)
data_vect = torch.from_numpy(np.concatenate([data_mean, data_std], axis=0))
one_hot = np.zeros(5)
one_hot[data_label] = 1
target = torch.from_numpy(one_hot)
print(data_vect)
print(one_hot)
print(target)
