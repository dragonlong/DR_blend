import argparse
import os
import shutil
import time
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from classify_loader import data_loader
from inception_v3 import inception_v3 as icp_v3_new


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    #choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=10000000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--ng-weights', type=float, default=0.1)

class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet') :
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(512, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else :
            raise("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False


    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y

class TrainingModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(TrainingModel, self).__init__()
        self.num_classes = num_classes
        self.upsampling = nn.modules.upsampling.Upsample(scale_factor=3)
        self.softmax = nn.Softmax()
        if arch.startswith('alexnet') :
            self.features = original_model.features  #TODO why input same but output different
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                #nn.Linear(12544, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet18') or arch.startswith('resnet34') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])  #jeff
            self.classifier = nn.Sequential(  #jeff
                #nn.Linear(512, num_classes)
                nn.Linear(18432, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('resnet50') or arch.startswith('resnet101') or arch.startswith('resnet152') :
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])  #jeff
            self.classifier = nn.Sequential(  #jeff
                nn.Linear(8192, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                #nn.Linear(25088, 4096),
                #nn.Linear(18432, 4096),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        elif arch.startswith('vgg19'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg19'
        elif arch.startswith('densenet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
               #nn.Linear(50176, num_classes) #121
               #nn.Linear(81536, num_classes) #169
               #nn.Linear(94080, num_classes) #201
               #nn.Linear(1920*6*6, num_classes)

               #nn.Linear(147456, num_classes)
               #nn.Linear(65536, num_classes)  #121 256
               #nn.Linear(50176, num_classes) #169 256
               nn.Linear(81536, num_classes) #169 512

            )
            self.modelName = 'densenet'
        elif arch.startswith('inception'):
            self.features = original_model

            self.classifier = nn.Sequential(
                #nn.Linear(106496, num_classes
            )  # 169 256

            self.modelName = 'inception'
        else :
            raise("Training not supported on this architecture yet")

       # Freeze those weights
       #for p in self.features.parameters():
       #    p.requires_grad = False
    def forward(self, x):
        #print(x.size())
        #x = self.upsampling(x)
        f = self.features(x)
        #print(f.size())
        if self.modelName == 'alexnet' :
            f = f.view(f.size(0), 256 * 6 * 6)
            #f = f.view(f.size(0), 12544)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'vgg19':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet' :
            #f = self.middle(f)  #jeff
            f = f.view(f.size(0), -1)
        elif self.modelName == 'densenet':
            f = f.view(f.size(0), -1)
        #print(f.size())
        y = self.classifier(f) #jeff
        #print(y.shape)
        #y = self.softmax(y)
        return y


def main():
    global args, best_loss
    best_loss = sys.float_info.max

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    #traindir = os.path.join(args.data, 'train')
    #valdir = os.path.join(args.data, 'val')
    # Get number of classes from train directory
    #num_classes = len([name for name in os：、.listdir(traindir)])
    num_classes = 5
    print("num_classes = '{}'".format(num_classes))
    # create model
    if args.finetune:
        print("=> using pre-trained model '{}'".format(args.arch))
        original_model = models.__dict__[args.arch](pretrained=True)
        model = FineTuneModel(original_model, args.arch, num_classes)
    else:
        if args.arch == 'inception_v3_new':
            original_model = icp_v3_new(pretrained=False, num_classes=num_classes)
            model = TrainingModel(original_model, args.arch, num_classes)
        else:
            print("=> creating model '{}'".format(args.arch))
            #model = models.__dict__[args.arch]()
            original_model = models.__dict__[args.arch](pretrained=False, num_classes= num_classes)
            model = TrainingModel(original_model, args.arch, num_classes)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}  ##num_workers
    # Data loading code
    #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        data_loader('/data/jeffery/kg/sample_512/', True, transform=
                    transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize((-257.478639,), (471.683592,)),
                    ])
        ),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        data_loader('/data/jeffery/kg/sample_512/', False, transform=
        transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize((-257.478639,), (471.683592,)),
        ])
                    ),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # define loss function (criterion) and pptimizer
    weight = torch.ones(num_classes)
    weight[0] = args.ng_weights
    #criterion = nn.CrossEntropyLoss(weight.cuda()).cuda()
    #criterion = torch.nn.MultiMarginLoss(weight = weight.cuda()).cuda()
    criterion = torch.nn.MultiMarginLoss().cuda()
    #criterion = torch.nn.CrossEntropyLoss(weight.cuda()).cuda()
    #criterion = torch.nn.CrossEntropyLoss().cuda()
    #criterion = torch.nn.MSELoss().cuda()
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), # Only finetunable params
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.evaluate:
        validate(test_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        loss = validate(test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        #is_best = prec1 > best_prec1
        #best_prec1 = max(prec1, best_prec1)
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    #prec = AverageMeter()
    #top1 = AverageMeter()
    #top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda(async=True)
        #print(target.numpy())
        #target_var = torch.autograd.Variable(target.type(torch.LongTensor)).cuda(async=True)
        target_var = torch.autograd.Variable(target.type(torch.FloatTensor)).cuda(async=True)
        #print(type(target_var.data))
        # compute output
        output = model(input_var)
        #print ('output', output.size())
        #print ('target', target_var.size())
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target)
        #prec_cur = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))
        #prec.update(prec_cur[0], input.size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Train Epoch: [{}][{}/{}]\t'
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'
                  #'Prec {prec.val:.3f}'
                  #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                .format(
                   epoch, i*len(input), len(train_loader.dataset),
                   #batch_time=batch_time,
                   #data_time=data_time,
                   loss=losses
                  # prec = prec
                #top1=top1, top5=top5

            ))

def accu_for_cls(predict, target, cls):
    target_np = target.data.numpy().cpu()
    predict_np = predict.data.numpy().cpu()
    correct = 0
    total = 0
    for i in range(len(target_np)):
        if target_np[i] == np.array(cls):
            if target_np[i] == predict_np[i]:
                correct += 1
            total += 1
    if total != 0:
        return correct/ total, total
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
    #top1 = AverageMeter()
    #top5 = AverageMeter()
    #prec = AverageMeter

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        #target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True).cuda(async=True)
        #target_var = torch.autograd.Variable(target.type(torch.LongTensor), volatile=True).cuda(async=True)
        target_var = torch.autograd.Variable(target.type(torch.FloatTensor), volatile=True).cuda(async=True)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        tpr = true_positive(output.data, target_var.data)
        acc = accuracy(output.data, target_var.data)
        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        tprs.update(tpr, input.size(0))
        accs.update(acc, input.size(0))
        #top1.update(prec1[0], input.size(0))
        #top5.update(prec5[0], input.size(0))
        #prec.update(prec[0], input.size(0))
        acc_0, num_0 = accu_for_cls(output.data, target_var.data, [1,0,0,0,0])
        acc_1, num_1 = accu_for_cls(output.data, target_var.data, [0,1,0,0,0])
        acc_2, num_2 = accu_for_cls(output.data, target_var.data, [0,0,1,0,0])
        acc_3, num_3 = accu_for_cls(output.data, target_var.data, [0,0,0,1,0])
        acc_4, num_4 = accu_for_cls(output.data, target_var.data, [0,0,0,0,1])
        accs_0.update(acc_0, num_0)
        accs_1.update(acc_1, num_1)
        accs_2.update(acc_2, num_2)
        accs_3.update(acc_3, num_3)
        accs_4.update(acc_4, num_4)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: [Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                  'tpr {tpr.val:.4f} ({tpr.avg:.4f})'
                  #'Prec {prec.val:.3f}'
                  #'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  #'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
            .format(
                   batch_time=batch_time, loss=losses,
                   acc = accs, tpr = tprs
                   #top1=top1, top5=top5
                   #prec = prec
                 ))

    #print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
    #      .format(top1=top1, top5=top5))

    #return top1.avg
    return losses.avg


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
    #print(target)
    _, o = torch.max(output, 1) #get the maximum idx in the second dim
    correct = o.eq(target)
    correct = correct.sum()
    return correct/ (output.size(0) + 1)


def true_positive(output, target):
    #print(output)
    _, o = torch.max(output, 1)
    #print(o)
    tp = target[o].sum()  #get the item from target when o == 1
    #print(target[o])
    return tp/(o.sum() + 1)


if __name__ == '__main__':
    main()