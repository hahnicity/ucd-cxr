# encoding: utf-8
#
# Code shamelessly taken from https://github.com/arnoweng/CheXNet

"""
The main CheXNet model implementation.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs


CKPT_PATH = 'model.pth.tar'
N_CLASSES = 14
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = '/data/datasets/chestxray14/images'
TRAIN_IMAGE_LIST = '/data/datasets/chestxray14/labels/train_list.processed'
TEST_IMAGE_LIST = '/data/datasets/chestxray14/labels/test_list.processed'
# XXX validation image list
BATCH_SIZE = 64


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model).cuda()

    if os.path.isfile(CKPT_PATH) and not args.train:
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    if args.train:
        run_model_training(model, normalize, args.epochs)
    run_model_testing(model, normalize)


def run_model_training(model, normalize, epochs):
    train_dataset = ChestXrayDataSet(
        data_dir=DATA_DIR,
        image_list_file=TRAIN_IMAGE_LIST,
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda
            (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda
            (lambda crops: torch.stack([normalize(crop) for crop in crops]))
        ]))

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=12, pin_memory=True
    )
    # XXX validation loaders

    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCELoss()
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
    for i in range(epochs):
        scheduler.step()
        for i, (inp, target) in enumerate(train_loader):
            target = target.cuda()
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
            output = model(input_var)

            optimizer.zero_grad()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # XXX run validation and perform LR reduction scheduling
    return model


def run_model_testing(model, normalize):
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=12, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        # Oh. They're taking the mean of all the crops here. It's almost like
        # an ensemble in itself.
        output_mean = output.view(bs, n_crops, -1).mean(1)
        # and this line doesn't actually do anything besides concatenate to an
        # array.
        pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        # OK, this is an important detail, the model is pretrained with ImageNet
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == '__main__':
    main()
