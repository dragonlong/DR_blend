#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 22:54:46 2018

@author: Jeremy
"""
import numpy as np
import h5py
import pandas as pd
import os

def load_h5_data(base_path):
    data = np.array([]).reshape(0, 4096)
    for i in range(1, 414):
        if i < 10:
            path = os.path.join(base_path, 'IDRiD_00%s.h5'%i)
            f = h5py.File(path, 'r')
            d = f.get('value')
            d = np.array(d).reshape([1,-1])
            data = np.concatenate((data, d), axis=0)
        if (i >= 10) and (i < 100):
            path = os.path.join(base_path, 'IDRiD_0%s.h5'%i)
            f = h5py.File(path, 'r')
            d = f.get('value')
            d = np.array(d).reshape([1,-1])
            data = np.concatenate((data, d), axis=0)
        if i >= 100:
            path = os.path.join(base_path, 'IDRiD_%s.h5'%i)
            f = h5py.File(path, 'r')
            d = f.get('value')
            d = np.array(d).reshape([1,-1])
            data = np.concatenate((data, d), axis=0)
    labels = pd.read_csv('/Users/Jeremy/Desktop/ISBI_Challenge/Trace_Data_Training_Set/IDRiD_Training Set.csv')
    y_dr = labels['Retinopathy grade'].values.reshape([-1, 1])
    y_dme = labels['Risk of macular edema'].values.reshape([-1, 1])
    return data, y_dr, y_dme

if __name__ =="__main__":
    base_path = "/media/dragonx/DataStorage/result3/"
    load_h5_data()



