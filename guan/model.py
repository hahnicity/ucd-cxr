"""
Reproduce as much of Guan's work as we care to. For now the global branch is good
enough.
"""
from argparse import ArgumentParser
from datetime import datetime
import multiprocessing
import os
from time import time

import torch
import torchvision
from torchvision import transforms

from cxrlib import constants
from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs, Meter


class ResNet50(torch.nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)


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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
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
        # Oh. They're taking the mean of all the crops here. It's almost like
        # an ensemble in itself.
        output_mean = output.view(bs, n_crops, -1).mean(1)
        # and this line doesn't actually do anything besides concatenate to an
        # array.
        pred = torch.cat((pred, output), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(constants.CLASS_NAMES[i], AUROCs[i]))


def main():
    parser = ArgumentParser()
    parser.add_argument('--data-path', help='path to dataset', default='/data/datasets/chestxray14/')
    parser.add_argument('--epochs', type=int, help='number of epochs to train model for', default=50)
    parser.add_argument('--batch-size', type=int, help='batch size of the global branch', default=128)
    parser.add_argument('--lr-decay-epochs', type=int, default=20, help='number of epochs before we decay the learning rate')
    parser.add_argument('--print-progress', action='store_true', help='flag to use if you want to track the models training progress')
    args = parser.parse_args()

    if args.print_progress:
        print("Model start time: {}".format(datetime.now().strftime("%Y-%m-%d_%H%M")))
    model = ResNet50().cuda()
    model = torch.nn.DataParallel(model).cuda()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    training_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    model, train_loss = train(model, training_transformations, args)
    model.save_state_dict('guan_global_{}.pt'.format(datetime.now().strftime("%Y_%m_%d_%H%M")))
    if args.print_progress:
        print("Model end train time: {}".format(datetime.now().strftime("%Y-%m-%d_%H%M")))

    validation_transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test(model, validation_transformations, args)


if __name__ == "__main__":
    main()
