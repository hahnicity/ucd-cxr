import argparse
import multiprocessing
import os
import sys

import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50

from cxrlib.init import student_t_init
from cxrlib.read_data import get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModelWithTestAUCReporting


def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('images_path')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--no-validation', action='store_true')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    # model hyperparameters
    args = parser.parse_args()

    initializer = lambda m: student_t_init(m, v=sys.maxsize)

    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = resnet50(pretrained=False, num_classes=14)
    model.apply(initializer)
    model = cuda_wrapper(torch.nn.DataParallel(model))

    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False

    if args.no_validation:
        train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed)
        valid_loader = None
    else:
        train_loader, valid_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, get_validation_set=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4, nesterov=True)
    lr_scheduler = None
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'model', False)
    runner = RunModelWithTestAUCReporting(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
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
    reporting.save_all('student_t_init_resnet_rgb')


if __name__ == "__main__":
    main()
