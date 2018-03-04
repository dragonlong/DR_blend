from datetime import datetime
import pprint
import os
import argparse
import numpy as np
import torchvision.models as models
from utils.util import mkdir
from data import FEATURE_DIR

mkdir(FEATURE_DIR)

class Config(object):
    def __init__(self, layers, cnf=None):
        self.layers = layersGlobalPoolLayer
        self.cnf = cnf
        pprint.pprint(cnf)

    def get(self, k, default=None):
        return self.cnf.get(k, default)

    @property
    def weights_epoch(self):
        path = "weights/{}/epochs".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    @property
    def weights_best(self):
        path = "weights/{}/best".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, '{epoch}_{timestamp}_{loss}.pkl')

    @property
    def weights_file(self):
        path = "weights/{}".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    @property
    def retrain_weights_file(self):
        path = "weights/{}/retrain".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights.pkl')

    @property
    def final_weights_file(self):
        path = "weights/{}".format(self.cnf['name'])
        mkdir(path)
        return os.path.join(path, 'weights_final.pkl')

    def get_features_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_mean_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'),  n_iter, skip)
        return os.path.join(FEATURE_DIR, fname)

    def get_std_fname(self, n_iter, skip=0, test=False):
        fname = '{}_{}_std_iter_{}_skip_{}.npy'.format(
            self.cnf['name'], ('test' if test else 'train'), n_iter, skip)
        return os.path.join(FEATURE_DIR, fname)

    def save_features(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_features_fname(n_iter, skip=skip,
                                              test=test), 'wb'), X)

    def save_std(self, X, n_iter, skip=0, test=False):
        np.save(open(self.get_std_fname(n_iter, skip=skip,
                                        test=test), 'wb'), X)

    def load_features(self, test=False):
        return np.load(open(self.get_features_fname(test=test)))


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


def para_config():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=10000000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
    parser.add_argument('--finetune', dest='finetune', action='store_true', help='fine tune pre-trained model')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--ng-weights', type=float, default=0.1)
    parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
    parser.add_argument('--display_single_pane_ncols', type=int, default=0,
                        help='if positive, display all images in a single visdom web panel with certain number of images per row.')
    parser.add_argument('--update_html_freq', type=int, default=1000,
                        help='frequency of saving training results to html')
    parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
    parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
    parser.add_argument('--save_epoch_freq', type=int, default=5,
                        help='frequency of saving checkpoints at the end of epochs')
    parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
    parser.add_argument('--epoch_count', type=int, default=1,
                        help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
    parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
    parser.add_argument('--which_epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')
    parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=100,
                        help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    # parser.add_argument('--lr', type=float, default=0.02, help='initial learning rate for adam')
    parser.add_argument('--no_lsgan', action='store_true',
                        help='do *not* use least square GAN, if false, use vanilla GAN')
    parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
    parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
    parser.add_argument('--pool_size', type=int, default=50,
                        help='the size of image buffer that stores previously generated images')
    parser.add_argument('--no_html', action='store_true',
                        help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
    parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
    parser.add_argument('--lr_decay_iters', type=int, default=50,
                        help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--identity', type=float, default=0.5,
                        help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
    parser.add_argument('--isTrain', default=True, help='decide whether training and plot')
    parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
    parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
    parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
    parser.add_argument('--name', type=str, default='experiment_name',
                        help='name of the experiment. It decides where to store samples and models')
    return parser
