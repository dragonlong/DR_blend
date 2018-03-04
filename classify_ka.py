import argparse
import os
import shutil
import time
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from classify_loader import data_loader
#from RAM_V0_512_4_4_32 import c_512_4_4_32
from RAM_512 import c_512_5_3_32
import utils.util as util
from utils import kappa

# The goal here is to visualize loss and RAM image with original image
# visualization demo with visdom
# import visdom
# vis = visdom.Visdom()
# vis.text('Hello, world!')
# vis.image(np.ones((3, 10, 10)))

# using visualizer function
from utils.visualizer import Visualizer


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000000, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',   help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,           metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,  metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',  help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,  metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,          metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true',  help='fine tune pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False,     help='disables CUDA training')
parser.add_argument('--ng-weights', type=float, default=0.1)
parser.add_argument('--display_freq', type=int, default=50, help='frequency of showing training results on screen')
parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
# parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--identity', type=float, default=0.5, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
parser.add_argument('--isTrain', default=True, help='decide whether training and plot')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

def main():
    global args, best_loss
    best_loss = sys.float_info.max

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #model = c_512_4_4_32()
    model = c_512_5_3_32()
    model = torch.nn.DataParallel(model).cuda()
    visualizer = Visualizer(args)

    cudnn.benchmark = True
    kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}  ##num_workers

    train_loader = torch.utils.data.DataLoader(
        data_loader('/media/dragonx/752d26ef-8f47-416d-b311-66c6dfabf4a3/sample_512/', True, transform=
        transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-257.478639,), (471.683592,)),
        ])
                    ),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        data_loader('/media/dragonx/752d26ef-8f47-416d-b311-66c6dfabf4a3/sample_512/', False, transform=
        transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((-257.478639,), (471.683592,)),
        ])
                    ),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),  # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.evaluate:
        validate(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, visualizer)

        # evaluate on validation set
        visualizer.reset()
        loss, score = validate(test_loader, model, criterion)
        errors_val = {}
        errors_val['val_loss'] = loss
        errors_val['val_accu'] = score
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
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # reset the data visualizer
        visualizer.reset()

        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        # print(target.numpy())
        #target_var = torch.autograd.Variable(target.type(torch.LongTensor)).cuda(async=True)
        target_var = torch.autograd.Variable(target.type(torch.FloatTensor)).cuda()

        # compute output
        output, ram_x , last_conv = model(input_var)
        # by doing element-wise comparison, we could make the regression more stable
        tensor_sub = output - target_var
        mask = tensor_sub.abs_() > 0.5
        t = (mask==0).nonzero()
        output[t, 0] = target_var[t, 0]
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target)
        # prec_cur = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        # prec.update(prec_cur[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})      \t'
        # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\

        # update the infos
        # if losses.count % args.display_freq == 0:
        #     save_result = total_steps % opt.update_html_freq == 0
        #     visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
        total_steps = losses.count
        orig_img = util.tensor2im(input)
        visuals = {}
        visuals['orig_img'] = orig_img
        average = torch.mean(torch.mean(last_conv, dim=2, keepdim=True), dim=3, keepdim=True)
        dimension = list(average.shape)
        # if dimension[0]== args.batch_size:
        #     t = torch.zeros(args.batch_size, 256, 1, 1)
        # else:
        t = torch.zeros(dimension[0], 256, 1, 1)
        weights = torch.addcdiv(t, 1.0, ram_x.data.cpu(), average.data.cpu())
        mul = torch.addcmul(t, 1.0, weights, last_conv.data.cpu())
        ram = torch.mean(mul, dim=1,keepdim=True)
        ram_single = ram[0]
        sub_predict = np.abs(target.numpy() - output.data.cpu().float().numpy())
        correct_predict = np.count_nonzero((sub_predict<0.5))/args.batch_size
        #score = kappa(target, output.data.cpu().float().numpy().astype(int))
        score = correct_predict

        visuals['RAM'] = util.tensor2im(ram)
        if total_steps % args.display_freq == 0:
            save_result = total_steps % args.update_html_freq == 0
            visualizer.display_current_results(visuals, i, save_result)

        # print training loss
        if i % args.print_freq == 0:
            print('Train Step: [{}][{}/{}]\t'
                  'Loss {loss.val:.10f} ({loss.avg:.10f})\n'
                # 'Prec {prec.val:.3f}'
                # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                .format(
                epoch, i * len(input), len(train_loader.dataset),
                # batch_time=batch_time,
                # data_time=data_time,
                loss=losses
                # prec = prec
                # top1=top1, top5=top5

            ))
            errors = {}
            errors["loss"] = losses.val
            errors["accuracy"] = score
            visualizer.plot_current_errors(epoch, float(i)*args.batch_size/35126, args, errors)
            #print_current_errors(self, epoch, i, errors, t)
            #plot_current_errors(self, epoch, counter_ratio, opt, errors)


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
    tprs = AverageMeter()
    accs = AverageMeter()
    accs_0 = AverageMeter()
    accs_1 = AverageMeter()
    accs_2 = AverageMeter()
    accs_3 = AverageMeter()
    accs_4 = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    # prec = AverageMeter

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda()
        #target_var = torch.autograd.Variable(target.type(torch.LongTensor), volatile=True).cuda(async=True)
        target_var = torch.autograd.Variable(target.type(torch.FloatTensor), volatile=True).cuda()
        # compute output
        output, _, _ = model(input_var)
        tensor_sub = output - target_var
        mask = tensor_sub.abs_() > 0.5
        t = (mask==0).nonzero()
        output[t, 0] = target_var[t, 0]
        loss = criterion(output, target_var)
        sub_predict = np.abs(target.numpy() - output.data.cpu().float().numpy())
        correct_predict = np.count_nonzero((sub_predict<0.5))/args.batch_size
        #score = kappa(target, output.data.cpu().float().numpy().astype(int))
        score = correct_predict
        # tpr = true_positive(output.data, target_var.data)
       # acc = accuracy(output.data, target_var.data)
        # measure accuracy and record loss
        # prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        accs.update(score, input.size(0))
        # tprs.update(tpr, input.size(0))
       # accs.update(acc, input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top5.update(prec5[0], input.size(0))
        # prec.update(prec[0], input.size(0))
        '''
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
        '''
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: [Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
          'tpr {tpr.val:.4f} ({tpr.avg:.4f})\t'
    # 'Prec {prec.val:.3f}'
    # 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
    # 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
          #'accs_0 {accs_0.val:.4f} ({accs_0.avg:.4f})\t'
          #'accs_1 {accs_1.val:.4f} ({accs_1.avg:.4f})\t'
          #'accs_2 {accs_2.val:.4f} ({accs_2.avg:.4f})\t'
          #'accs_3 {accs_3.val:.4f} ({accs_3.avg:.4f})\t'
          #'accs_4 {accs_4.val:.4f} ({accs_4.avg:.4f})\t'
        .format(
        batch_time=batch_time, loss=losses,
        acc=accs, tpr=tprs,
        # top1=top1, top5=top5
        # prec = prec
        #accs_0=accs_0, accs_1=accs_1, accs_2=accs_2, accs_3=accs_3, accs_4=accs_4

    ))

    # print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #      .format(top1=top1, top5=top5))

    # return top1.avg

    return losses.avg, accs.avg



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr


'''
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
'''
'''
def accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    #truth = target >= 0.5
    truth = target.data().cpu()
    acc = pred.eq(truth).sum() / target.numel()
    return acc
'''


def accuracy(output, target):
    # print(target)
    _, o = torch.max(output, 1)  # get the maximum idx in the second dim
    correct = o.eq(target)
    correct = correct.sum()
    return correct / (output.size(0) + 1)


def true_positive(output, target):
    # print(output)
    _, o = torch.max(output, 1)
    # print(o)
    tp = target[o].sum()  # get the item from target when o == 1
    # print(target[o])
    return tp / (o.sum() + 1)


if __name__ == '__main__':
    main()
