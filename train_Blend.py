import argparse
import os
import shutil
import time
import sys
import torch
import numpy as np
from torch import nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from classify_loader import FeatureLoader
from configs.c_blend_fc import blendNet
from utils.util import kappa
from config import para_config

# using visualizer function
from utils.visualizer import Visualizer
# parameters setup
parser = para_config()


def main():
    global args, best_loss
    best_loss = sys.float_info.max

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    model = blendNet()
    model = torch.nn.DataParallel(model).cuda()
    visualizer = Visualizer(args)

    cudnn.benchmark = True
    kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}  ##num_workers

    train_loader = torch.utils.data.DataLoader(
        FeatureLoader('/media/dragonx/DataStorage/download/5label_mean_std/', True,transform=
        transforms.Compose([
            transforms.ToTensor(),
        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        FeatureLoader('/media/dragonx/DataStorage/download/5label_mean_std/', False, transform=
        transforms.Compose([
            transforms.ToTensor(),

        ])
                    ),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),  # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
   # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.evaluate:
        validate(test_loader, model, criterion)
        return
    errors_val_set = []
    errors_val = {}
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, visualizer)
        # evaluate on validation set
        visualizer.reset()
        loss, score = validate(test_loader, model, criterion)
        errors_val['val_loss'] = loss
        errors_val['val_accu'] = score
        errors_val_set.append(errors_val)
        print('evaluation loss is %f at epoch %d' %(loss, epoch))
        # visualizer.plot_current_errors(epoch, float(i)*args.batch_size/35126, args, errors)
        # visualizer.plot_current_errors(epoch, 0, args, errors_val)
        # remember best prec@1 and save checkpoint
        # is_best = prec1 > best_prec1
        # best_prec1 = max(prec1, best_prec1)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch, args, visualizer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (data_input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # reset the data visualizer
        visualizer.reset()

        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(data_input.type(torch.FloatTensor)).cuda()
        target_var = torch.autograd.Variable(target.type(torch.LongTensor)).cuda()

        # compute output
        output = model(input_var)
        loss = criterion(output, torch.squeeze(target_var))

        # measure accuracy and record loss
        losses.update(loss.data[0], data_input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # update the infos
        # total_steps = losses.count
        acc = accuracy(output.data, target_var.data)
        # if total_steps % args.display_freq == 0:
        #     save_result = total_steps % args.update_html_freq == 0
        #     visualizer.display_current_results(visuals, i, save_result)

        # print training loss
        errors = {}
        if i % args.print_freq == 0:
            print('Train Step: [{}][{}/{}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})\n'
                .format(epoch, i * len(data_input), len(train_loader.dataset), loss=losses
            ))
            errors["loss"] = losses.val
            errors["accuracy"] = acc
            visualizer.plot_current_errors(epoch, float(i)*args.batch_size/(35126*0.8), args, errors)


def accu_for_cls(predict, target, cls):
    target_np = target.cpu().numpy()
    predict_np = predict.cpu().numpy()
    correct = 0
    total = 0
    for i in range(len(target_np)):
        if target_np[i] == cls:
            if target_np[i] == np.argmax(predict_np[i], axis=0):
                correct += 1
            total += 1
    if total != 0:
        return correct / total, total
    else:
        return 0, 0


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    accs_0 = AverageMeter()
    accs_1 = AverageMeter()
    accs_2 = AverageMeter()
    accs_3 = AverageMeter()
    accs_4 = AverageMeter()
    cfm = ConfusionMeter(5)
    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (data_input, target) in enumerate(val_loader):
        #async=True
        input_var = torch.autograd.Variable(data_input.type(torch.FloatTensor), volatile=True).cuda()
        target_var = torch.autograd.Variable(target.type(torch.LongTensor), volatile=True).cuda()
        output = model(input_var)
        loss = criterion(output, torch.squeeze(target_var))
        acc = accuracy(output.data, target_var.data)
        # measure accuracy and record loss
        losses.update(loss.data[0], data_input.size(0))
        accs.update(acc, data_input.size(0))

        acc_0, num_0 = accu_for_cls(output.data, target_var.data, 0)
        acc_1, num_1 = accu_for_cls(output.data, target_var.data, 1)
        acc_2, num_2 = accu_for_cls(output.data, target_var.data, 2)
        acc_3, num_3 = accu_for_cls(output.data, target_var.data, 3)
        acc_4, num_4 = accu_for_cls(output.data, target_var.data, 4)
        accs_0.update(acc_0, num_0)
        accs_1.update(acc_1, num_1)
        accs_2.update(acc_2, num_2)
        accs_3.update(acc_3, num_3)
        accs_4.update(acc_4, num_4)
        #cfm.add(output, target_var)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
   # print("confusion matrix: ", cfm.conf)
    return losses.avg, accs.avg


class ConfusionMeter():
    """
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    """

    def __init__(self, k, normalized=False):
        """
        Args:
            k (int): number of classes in the classification problem
            normalized (boolean): Determines whether or not the confusion matrix
                is normalized or not
        """
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        predicted = predicted.cpu().data.numpy()
        target = target.cpu().data.numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


def accuracy(output, target):
    # print(target)
    _, o = torch.max(output, 1)  # get the maximum idx in the second dim
    correct = o.eq(torch.squeeze(target))
    correct = correct.sum()
    return correct / (output.size(0))


def true_positive(output, target):
    # print(output)
    _, o = torch.max(output, 1)
    # print(o)
    tp = target[o].sum()  # get the item from target when o == 1
    # print(target[o])
    return tp / (o.sum() + 1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
