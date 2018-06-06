"""
Reproduce as much of Guan's work as we care to. For now the global branch is good
enough.
"""
from argparse import ArgumentParser
from datetime import datetime
import multiprocessing
import os
from time import time

import numpy as np
import torch
import torchvision
from torchvision import transforms

from cxrlib import constants
from cxrlib.models import GuanResNet50
from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs, Meter


def train(model, transformations, args):
    train_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.data_path, "images"),
        image_list_file=os.path.join(args.data_path, "labels", "train_val_list.processed"),
        transform=transformations
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
    )
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs)
    train_loss = Meter('loss')

    for ep in range(args.epochs):
        scheduler.step()
        for i, (inp, target) in enumerate(train_loader):
            start_batch = time()
            target = target.cuda()
            bs, c, h, w = inp.size()
            target = torch.autograd.Variable(target)
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
            output = model(input_var)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            batch_time = time() - start_batch
            train_loss.update(loss)
            if args.print_progress:
                print('Epoch: {}/{}, Batch: {}/{}, Batch time: {}, Loss: {}\r'.format(
                    ep+1, args.epochs, i, len(train_loader), batch_time, str(train_loss)
                ), end="")
        if args.print_progress:
            print("")
        print("end epoch {}".format(ep))
    return model, train_loss


def test(model, transformations, args):
    test_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.data_path, "images"),
        image_list_file=os.path.join(args.data_path, "labels", "test_list.processed"),
        transform=transformations
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
    )
    model.eval()

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        # The .data method converts from Variable to Tensor. And you can only
        # concatenate a tensor with a tensor, not variable with a tensor.
        pred = torch.cat((pred, output.data), 0)
        print('Batch: {}/{}\r'.format(i, len(test_loader)), end="")

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(len(constants.CLASS_NAMES)):
        print('The AUROC of {} is {}'.format(constants.CLASS_NAMES[i], AUROCs[i]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--load-weights', help='path to file of weights to load')
    parser.add_argument('--data-path', help='path to dataset', default='/fastdata/chestxray14/')
    parser.add_argument('--epochs', type=int, help='number of epochs to train model for', default=50)
    parser.add_argument('--batch-size', type=int, help='batch size of the global branch', default=128)
    parser.add_argument('--lr-decay-epochs', type=int, default=20, help='number of epochs before we decay the learning rate')
    parser.add_argument('--print-progress', action='store_true', help='flag to use if you want to track the models training progress')
    args = parser.parse_args()

    if args.print_progress:
        print("Model start time: {}".format(datetime.now().strftime("%Y-%m-%d_%H%M")))
    model = GuanResNet50().cuda()
    model = torch.nn.DataParallel(model).cuda()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    if not args.load_weights:
        training_transformations = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        model, train_loss = train(model, training_transformations, args)
        torch.save(model.module.state_dict(), 'guan_global_{}.pt'.format(datetime.now().strftime("%Y_%m_%d_%H%M")))
        if args.print_progress:
            print("Model end train time: {}".format(datetime.now().strftime("%Y-%m-%d_%H%M")))
    else:
        model_weights = torch.load(args.load_weights)
        model.module.load_state_dict(model_weights)

    validation_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test(model, validation_transformations, args)


if __name__ == "__main__":
    main()
