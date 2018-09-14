from __future__ import print_function

import os
import argparse

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from datagen import ListDataset
from loss import BasicLoss, FocalLoss, IoULoss, StatLoss
from retinanet import RetinaNet
from utils import kaggle_iou


# XXX for now exclude test eval
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('image_dir', help='Directory for stored images')
parser.add_argument('--train-list', help='Path to file with train image labels', default='rsna-train.csv')
parser.add_argument('--val-list', help='Path to file with val image labels', default='rsna-val.csv')
#parser.add_argument('test_list', help='Path to file with test image labels')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('-s', '--save-to', default='ckpt.pth')
parser.add_argument('-l', '--load-from', default='ckpt.pth')
parser.add_argument('-b', '--batch-size', default=16, type=int)
args = parser.parse_args()

assert torch.cuda.is_available(), 'Error: CUDA not found!'
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

trainset = ListDataset(root=args.image_dir,
                       list_file=args.train_list, train=True, transform=transform, input_size=224)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

testset = ListDataset(root=args.image_dir,
                      list_file=args.val_list, train=False, transform=transform, input_size=224, val=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)


# Model
net = RetinaNet(num_classes=1)
net.load_state_dict(torch.load('./model/guan.pth'))
best_perf = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/{}'.format(args.load_from))
    net.load_state_dict(checkpoint['net'])
    best_perf = checkpoint['perf']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

#criterion = FocalLoss(num_classes=1)
criterion = StatLoss()
#criterion = IoULoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    net.module.freeze_bn()
    train_loss = 0
    print("")
    for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
        inputs = Variable(inputs).cuda()
        loc_targets = Variable(loc_targets).cuda()
        cls_targets = Variable(cls_targets).cuda()
        optimizer.zero_grad()
        loc_preds, cls_preds = net(inputs)
        # XXX It might be interesting to try alternating loss funcs. Like one epoch you
        # do just vanilla basicloss, and the next you do your modified, faster variant.
        loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
        loss.backward()
        optimizer.step()

        train_loss += loss
        print('train_loss: %.3f | avg_loss: %.3f\r' % (loss.data[0], train_loss/(batch_idx+1)), end="")

# Test
def test(epoch, best_perf):
    print('\nTest')
    net.eval()
    test_loss = 0
    print("")
    kaggle_sum_ap = 0
    kaggle_div = 0
    perf_reasons = None

    with torch.no_grad():
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = Variable(inputs).cuda()
            loc_targets = Variable(loc_targets).cuda()
            cls_targets = Variable(cls_targets).cuda()
            loc_preds, cls_preds = net(inputs)
            tmp_ap, tmp_div, tmp_perf_reasons = kaggle_iou(loc_preds, cls_preds, loc_targets, cls_targets)
            kaggle_sum_ap += tmp_ap
            kaggle_div += tmp_div
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss
            print('test_loss: %.3f | avg_loss: %.3f\r' % (loss.data[0], test_loss/(batch_idx+1)), end="")
            if perf_reasons is None:
                perf_reasons = tmp_perf_reasons
            else:
                for key in tmp_perf_reasons:
                    perf_reasons[key] += tmp_perf_reasons[key]

    print('')
    print(perf_reasons)
    perf = kaggle_sum_ap / kaggle_div
    print('\nkaggle cur perf: {} best perf: {}'.format(perf, best_perf))
    # Save checkpoint
    test_loss /= len(testloader)
    if perf > best_perf:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'perf': perf,
            'epoch': epoch,
            'loss': test_loss,
            'perf_reasons': perf_reasons,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/{}'.format(args.save_to))
        best_perf = perf
    return best_perf


def alternating_loss_run(best_perf):
    for epoch in range(start_epoch, start_epoch+args.epochs):
        if isinstance(criterion, BasicLoss):
            criterion = StatLoss()
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

        else:
            criterion = BasicLoss()
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

        train(epoch)
        best_perf = test(epoch, best_perf)


def focal_loss_run(best_perf):
    for epoch in range(start_epoch, start_epoch+args.epochs):
        criterion = FocalLoss(num_classes=1)
        train(epoch)
        best_perf = test(epoch, best_perf)


#alternating_loss_run(best_perf)
focal_loss_run(best_perf)
