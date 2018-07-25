import argparse
import multiprocessing
import os

import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50

from cxrlib.models.bottleneck_attention import resnet50ish
from cxrlib.read_data import get_five_crop_loaders, get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel


def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('images_path')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--loader', choices=['five_crop', 'guan'], default='guan')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    # model hyperparameters
    parser.add_argument('-r', '--reduction-coef', default=16, type=int)
    parser.add_argument('-d', '--dilation', default=4, type=int)
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    # model is pretrained by default
    model = resnet50ish(args.reduction_coef, args.dilation)
    model = cuda_wrapper(torch.nn.DataParallel(model))

    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False

    if args.no_validation and args.loader == 'guan':
        train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed)
        valid_loader = None
    elif not args.no_validation and args.loader == 'guan':
        train_loader, valid_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, get_validation_set=True)
    elif args.no_validation and args.loader == 'five_crop':
        train_loader, test_loader = get_five_crop_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed)
        valid_loader = None
    else:
        train_loader, valid_loader, test_loader = get_five_crop_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, get_validation_set=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4, nesterov=True)
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'model', False)
    runner = RunModel(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        None,
        criterion,
        True if args.device == 'cuda' else False,
        reporting,
        validation_loader=valid_loader,
    )
    runner.train_multi_epoch(args.epochs)
    del train_loader
    del valid_loader
    torch.cuda.empty_cache()
    runner.generic_test_epoch()
    reporting.save_all('bam-resnet50ish-rgb')


if __name__ == "__main__":
    main()
