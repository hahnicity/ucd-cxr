import argparse
import multiprocessing
import os
from time import time

import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50

from cxrlib.init import kaiming_init, xavier_init
from cxrlib.read_data import get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel



def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('images_path')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results/loss_analysis'))
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--load-openi-model', help='specify path to stored openi model')
    parser.add_argument('--no-validation', action='store_true')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--weight-init', choices=['xavier', 'kaiming'], default='xavier')
    # model hyperparameters
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    if not args.load_openi_model and not args.pretrained:
        model = resnet50(pretrained=False, num_classes=14)
        if args.weight_init == 'kaiming':
            model.apply(kaiming_init)
        elif args.weight_init == 'xavier':
            model.apply(xavier_init)
        model = cuda_wrapper(torch.nn.DataParallel(model))
    elif args.pretrained:
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, 14)
        model = cuda_wrapper(torch.nn.DataParallel(model))
    else:
        model = torch.load(args.load_openi_model)
        model.module.fc = torch.nn.Linear(model.module.fc.in_features, 14)
        model = cuda_wrapper(model)

    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False

    if args.no_validation:
        train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed)
    else:
        train_loader, valid_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, get_validation_set=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    criterion = torch.nn.BCEWithLogitsLoss(reduce=False, size_average=False)
    reporting = Reporting(args.results_path)
    reporting.register(model, 'model', False)
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    cuda_async_wrapper = lambda x: x.cuda(non_blocking=True) if args.device == 'cuda' else x
    model.train()
    with torch.enable_grad():
        for epoch_num in range(args.epochs):
            for i, (inp, target) in enumerate(train_loader):

                batch_start = time()

                target = cuda_async_wrapper(target)
                inp = cuda_async_wrapper(inp)
                target = torch.autograd.Variable(target)
                # If we are performing multi-cropping squash the crops
                # into the batch we are using
                if inp.ndimension() == 5:
                    bs, crops, c, h, w = inp.size()
                    inp = inp.view(-1, c, h, w)
                    bs, crops, labs = target.size()
                    target = target.view(-1, labs)
                inp = torch.autograd.Variable(inp, requires_grad=True)
                output = model(inp)

                optimizer.zero_grad()
                loss = criterion(output, target)
                loss.mean().backward()
                optimizer.step()
                torch.save(
                    loss.detach().cpu().data,
                    os.path.join(args.results_path, "epoch-{}-batch-{}-loss.pt".format(epoch_num+1, i+1))
                )
                torch.save(
                    target.detach().cpu().data,
                    os.path.join(args.results_path, "epoch-{}-batch-{}-gt.pt".format(epoch_num+1, i+1))
                )
                del loss

                batch_time = round(time() - batch_start, 4)

                if args.print_progress:
                    print('Epoch: {}, Batch: {}/{}, Batch time: {}\r'.format(
                        epoch_num+1, i+1, len(train_loader), batch_time)
                    , end="")


if __name__ == "__main__":
    main()
