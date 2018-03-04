import numpy as np

from config import Config
from data_orig import BALANCE_WEIGHTS
from layers import *

cnf = {
    'name': __name__.split('.')[-1],
    'w': 224,
    'h': 224,
    'train_dir': 'data/train_small',
    'test_dir': 'data/test_small',
    'batch_size_train': 128,
    'batch_size_test': 16,
    'balance_weights': np.array(BALANCE_WEIGHTS),
    'balance_ratio': 0.975,
    'final_balance_weights':  np.array([1, 2, 2, 2, 2], dtype=float),
    'aug_params': {
        'zoom_range': (1 / 1.15, 1.15),
        'rotation_range': (0, 360),
        'shear_range': (0, 0),
        'translation_range': (-40, 40),
        'do_flip': True,
        'allow_stretch': True,
    },
    'weight_decay': 0.000,
    'sigma': 0.5,
    'schedule': {
        0: 0.001,
        150: 0.0001,
        201: 'stop',
    },
}

n = 32

layers = [
    (InputLayer, {'shape': (None, 3, cnf['w'], cnf['h'])}),
    ] + ([(ToCC, {})] if CC else []) + [
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(n, filter_size=(5, 5), stride=(2, 2))),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(n)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(2 * n, filter_size=(5, 5), stride=(2, 2))),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(2 * n)),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(2 * n)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(4 * n)),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(4 * n)),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(4 * n)),
    (BatchNormLayer, {}),
    (MaxPool2DLayer, pool_params()),
    (Conv2DLayer, conv_params(8 * n)),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(8 * n)),
    (BatchNormLayer, {}),
    (Conv2DLayer, conv_params(8 * n)),
    ] + ([(FromCC, {})] if CC else []) + [
    #(RMSPoolLayer, {'pool_size': (3, 3), 'stride':(3, 3)}),
    #(DropoutLayer, {'p': 0.5}),
    #(DenseLayer, dense_params(1024)),
    #(FeaturePoolLayer, {'pool_size': 2}),
    #(DropoutLayer, {'p': 0.5}),
    #(DenseLayer, dense_params(1024)),
    #(FeaturePoolLayer, {'pool_size': 2}),
    (GlobalPoolLayer,{}),
    (DenseLayer, {'num_units': 1}),
]

config = Config(layers=layers, cnf=cnf)
