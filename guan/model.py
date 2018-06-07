"""
Reproduce as much of Guan's work as we care to. For now the global branch is good
enough.
"""
from argparse import ArgumentParser
from copy import deepcopy
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
from cxrlib.results import compute_AUCs, Meter, SavedObjects
from cxrlib.swa import SWA


def train(model, saved_objects, args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.data_path, "images"),
        image_list_file=os.path.join(args.data_path, "labels", "train_val_list.processed"),
        transform=transformations
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
    )
    # Set model.training to True. This determines how batch norm and dropout perform
    model.train()

    if args.swa_start:
        swa_model = deepcopy(model)
        swa = SWA(model, swa_model, args.swa_start, train_loader, args.epochs, device='cuda')
        saved_objects.register(swa_model, "swa_weights", True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    criterion = torch.nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_decay_epochs)
    train_batch_loss = Meter('train_batch_loss')
    train_epoch_loss = Meter('train_epoch_loss')
    saved_objects.register(train_batch_loss, "train_batch_loss", False)
    saved_objects.register(train_epoch_loss, 'train_epoch_loss', False)
    if args.test_on_train_ep:
        test_auc = Meter('test_auc')
        saved_objects.register(test_auc, "test_auc", False)
    if args.test_on_train_ep and args.swa_start:
        swa_test_auc = Meter('swa_test_auc')
        saved_objects.register(swa_test_auc, 'swa_test_auc', False)

    for ep in range(args.epochs):
        scheduler.step()
        for i, (inp, target) in enumerate(train_loader):
            start_batch = time()
            target = target.cuda(async=True)
            inp = inp.cuda(async=True)
            bs, c, h, w = inp.size()
            target = torch.autograd.Variable(target)
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w))
            output = model(input_var)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_batch_loss.update(loss)
            if args.print_progress:
                batch_time = time() - start_batch
                print('Epoch: {}/{}, Batch: {}/{}, Batch time: {}, Loss: {}\r'.format(
                    ep+1, args.epochs, i, len(train_loader), batch_time, str(train_batch_loss)
                ), end="")

        train_epoch_loss.update(train_batch_loss.value())
        if args.print_progress:
            print("")
        if args.swa_start:
            swa.step()
        if args.test_on_train_ep:
            model_auc = np.array(test(model, args)).mean()
            test_auc.update(model_auc)
        if args.test_on_train_ep and args.swa_start and swa.is_swa_training():
            if ep < args.epochs - 1:
                swa.bn_update()
            swa_auc = np.array(test(swa_model, args)).mean()
            swa_test_auc.update(swa_auc)
        model.train()
        print("end epoch {}".format(ep))

    return model


def test(model, args):
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = ChestXrayDataSet(
        data_dir=os.path.join(args.data_path, "images"),
        image_list_file=os.path.join(args.data_path, "labels", "test_list.processed"),
        transform=transformations
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
    )

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
        if args.print_progress:
            print('Batch: {}/{}\r'.format(i, len(test_loader)), end="")

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    if args.print_results:
        print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
        for i in range(len(constants.CLASS_NAMES)):
            print('The AUROC of {} is {}'.format(constants.CLASS_NAMES[i], AUROCs[i]))

    return AUROCs


def main():
    parser = ArgumentParser()
    parser.add_argument('--load-weights', help='path to file of weights to load')
    parser.add_argument('--data-path', help='path to dataset', default='/fastdata/chestxray14/')
    parser.add_argument('--epochs', type=int, help='number of epochs to train model for', default=50)
    parser.add_argument('--batch-size', type=int, help='batch size of the global branch', default=128)
    parser.add_argument('--lr-decay-epochs', type=int, default=20, help='number of epochs before we decay the learning rate')
    parser.add_argument('--print-progress', action='store_true', help='flag to use if you want to track the models training progress')
    parser.add_argument('--swa-start', type=int, help='Run with Stochastic Weight Averaging and start SWA at a particular epoch')
    parser.add_argument('--test-on-train-ep', action='store_true', help='Test on a training epoch')
    parser.add_argument('--print-results', action='store_true', help='print results at the end of a test execution')
    args = parser.parse_args()

    print("Model start time: {}".format(datetime.now().strftime("%Y-%m-%d_%H%M")))
    saved_objs = SavedObjects(os.path.join(os.path.abspath(os.path.dirname(__file__)), "results"))
    model = GuanResNet50().cuda()
    model = torch.nn.DataParallel(model).cuda()
    saved_objs.register(model, "global_weights", True)

    if not args.load_weights:
        model = train(model, saved_objs, args)
        suffix = datetime.now().strftime("%Y_%m_%d_%H%M")
        saved_objs.save_all(suffix)
        print("Model end train time: {}".format(datetime.now().strftime("%Y-%m-%d_%H%M")))
    else:
        model_weights = torch.load(args.load_weights)
        model.module.load_state_dict(model_weights)

    if not args.test_on_train_ep:
        test(model, args)


if __name__ == "__main__":
    main()
