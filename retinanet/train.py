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

from retinanet.datagen import ListDataset
from retinanet.loss import BasicLoss, FocalLoss, IoULoss, StatLoss
from retinanet.retinanet_bbox import RetinaNet
from retinanet.utils import kaggle_iou


# XXX for now exclude test eval
parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('image_dir', help='Directory for stored images')
parser.add_argument('--train-list', help='Path to file with train image labels', default='rsna-train.csv')
parser.add_argument('--val-list', help='Path to file with val image labels', default='rsna-val.csv')
parser.add_argument('-i', '--initial-model', default='model/guan.pt', help='initial retinanet model to load')
#parser.add_argument('test_list', help='Path to file with test image labels')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('-s', '--save-to', default='ckpt.pth')
parser.add_argument('-l', '--load-from', default='checkpoint/ckpt.pth')
parser.add_argument('-b', '--batch-size', default=16, type=int)
parser.add_argument('--ask-to-save', action='store_true', help="ask to save a model after test results regardless of how high iou is")
parser.add_argument('--test-output-file', default='testing-results.txt')
parser.add_argument('--only-uni-or-bilateral', action='store_true', help='only train on unilateral or bilateral pneumonias')
parser.add_argument('--save-on', choices=['iou', 'loss'], help='save model based on test performance on iou or loss')
parser.add_argument('-u', '--undersample', type=float, help='undersample majority class with this ratio to the minority class')
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
                       list_file=args.train_list, train=True, transform=transform, input_size=224, only_uni_or_bilateral=args.only_uni_or_bilateral, undersample=args.undersample)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

testset = ListDataset(root=args.image_dir,
                      list_file=args.val_list, train=False, transform=transform, input_size=224, val=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, collate_fn=testset.collate_fn)


# Model
net = RetinaNet(num_classes=1)
net.load_state_dict(torch.load(args.initial_model))
best_iou = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(args.load_from)
    net.load_state_dict(checkpoint['net'])
    best_iou = checkpoint['perf']
    start_epoch = checkpoint['epoch']

net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.cuda()

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)

class RunModel(object):
    def __init__(self, net, trainloader, testloader, optimizer, criterion):
        self.best_iou = best_iou
        self.best_loss = best_loss
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.optimizer = optimizer
        self.criterion = criterion

    # Training
    def train(self, epoch):
        print('\nEpoch: %d' % epoch)
        self.net.train()
        self.net.module.freeze_bn()
        train_loss = 0
        print("")
        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.trainloader):
            inputs = Variable(inputs).cuda()
            loc_targets = Variable(loc_targets).cuda()
            cls_targets = Variable(cls_targets).cuda()
            self.optimizer.zero_grad()
            loc_preds, cls_preds = self.net(inputs)
            # XXX It might be interesting to try alternating loss funcs. Like one epoch you
            # do just vanilla basicloss, and the next you do your modified, faster variant.
            loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss
            print('train_loss: %.3f | avg_loss: %.3f\r' % (loss.data[0], train_loss/(batch_idx+1)), end="")

    # Test
    def test(self, epoch):
        print('\nTest')
        self.net.eval()
        test_loss = 0
        print("")
        kaggle_sum_ap = 0
        kaggle_div = 0
        perf_reasons = None

        with torch.no_grad():
            for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(self.testloader):
                inputs = Variable(inputs).cuda()
                loc_targets = Variable(loc_targets).cuda()
                cls_targets = Variable(cls_targets).cuda()
                loc_preds, cls_preds = self.net(inputs)
                tmp_ap, tmp_div, tmp_perf_reasons = kaggle_iou(loc_preds, cls_preds, loc_targets, cls_targets)
                kaggle_sum_ap += tmp_ap
                kaggle_div += tmp_div
                loss = self.criterion(loc_preds, loc_targets, cls_preds, cls_targets)
                test_loss += loss
                print('test_loss: %.3f | avg_loss: %.3f\r' % (loss.data[0], test_loss/(batch_idx+1)), end="")
                if perf_reasons is None:
                    perf_reasons = tmp_perf_reasons
                else:
                    for key in tmp_perf_reasons:
                        perf_reasons[key] += tmp_perf_reasons[key]

        test_loss = test_loss / (batch_idx+1)
        print('')
        print(perf_reasons)
        test_iou = kaggle_sum_ap / kaggle_div
        print('\nkaggle cur test_iou: {} best test_iou: {}'.format(test_iou, self.best_iou))
        # Save checkpoint
        test_loss /= len(self.testloader)
        if (test_iou > self.best_iou and args.save_on == 'iou') or (test_loss < self.best_loss and args.save_on == 'loss'):
            if test_loss < self.best_loss:
                self.best_loss = test_loss
            if test_iou > self.best_iou:
                self.best_iou = test_iou
            print('Saving..')
            state = {
                'net': self.net.module.state_dict(),
                'perf': test_iou,
                'epoch': epoch,
                'loss': test_loss,
                'perf_reasons': perf_reasons,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/{}'.format(args.save_to))

        if args.ask_to_save:
            should_save = input("should we save the classifier? [y/n] ")
            if should_save == 'y':
                state = {
                    'net': self.net.module.state_dict(),
                    'perf': test_iou,
                    'epoch': epoch,
                    'loss': test_loss,
                    'perf_reasons': perf_reasons,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, './checkpoint/{}'.format(args.save_to))

        with open(args.test_output_file, 'a') as results_output:
            results_output.write("epoch: {} loss: {} iou: {}\nreasons: {}\n".format(epoch, test_loss, test_iou, perf_reasons))


def stat_loss_run():
    criterion = StatLoss()
    runner = RunModel(net, trainloader, testloader, optimizer, criterion)
    for epoch in range(start_epoch, start_epoch+args.epochs):
        runner.train(epoch)
        runner.test(epoch)


# XXX do we even know if this works or not?
def alternating_loss_run():
    for epoch in range(start_epoch, start_epoch+args.epochs):
        if isinstance(criterion, BasicLoss):
            criterion = StatLoss()
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn, only_uni_or_bilateral=args.only_uni_or_bilateral)

        else:
            criterion = BasicLoss()
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)

        train(epoch)
        test(epoch)


def focal_loss_run():
    criterion = FocalLoss(num_classes=1)
    runner = RunModel(net, trainloader, testloader, optimizer, criterion)
    for epoch in range(start_epoch, start_epoch+args.epochs):
        runner.train(epoch)
        runner.test(epoch)


#alternating_loss_run()
focal_loss_run()
#stat_loss_run()
