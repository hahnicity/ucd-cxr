import argparse
import multiprocessing
import os

import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50

from cxrlib.init import kaiming_init, xavier_init
from cxrlib.read_data import get_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModelWithAUCAndValLR


def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('images_path')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--load-openi-model', help='specify path to stored openi model')
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--loader', choices=['baltruschat', 'five_crop', 'guan'], default='guan')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-init', choices=['xavier', 'kaiming'], default='kaiming')
    parser.add_argument('-r', '--run-test-after-epoch', type=int, help='run testing auc calculations after epoch N', default=0)
    # model hyperparameters
    parser.add_argument('-lr', '--loss-rate', type=float, default=.001)
    parser.add_argument('--nesterov', action='store_true')
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    if not args.load_openi_model and not args.pretrained:
        model = resnet50(pretrained=False, num_classes=14)
        if args.weight_init == 'xavier':
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
        train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, transform_type=args.loader)
        valid_loader = None
    else:
        train_loader, valid_loader, test_loader = get_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, get_validation_set=True, transform_type=args.loader)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.loss_rate, momentum=.9, weight_decay=1e-4, nesterov=args.nesterov)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path, 'resnet50-color-loader-{}-pretrained-{}-bs-{}-lr-{}-nesterov-{}'.format(args.loader, args.pretrained, args.batch_size, args.loss_rate, args.nesterov))
    reporting.register(model, 'model', False)
    runner = RunModelWithAUCAndValLR(
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
    reporting.save_all()


if __name__ == "__main__":
    main()
