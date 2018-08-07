import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.models.se_resnet import se_resnet50
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
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--loader', choices=['baltruschat', 'five_crop', 'guan'], default='guan')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('-r', '--run-test-after-epoch', type=int, help='run testing auc calculations after epoch N', default=0)
    # model hyperparameters
    parser.add_argument('-lr', '--loss-rate', type=float, default=.001)
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x

    model = se_resnet50(14)
    model = cuda_wrapper(torch.nn.DataParallel(model))

    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False

    if args.no_validation:
        train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, transform_type=args.loader)
        valid_loader = None
    else:
        train_loader, valid_loader, test_loader = get_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, get_validation_set=True, transform_type=args.loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.loss_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, mode='min')
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path, 'squeeze-excite-resnet50-rgb-loader-{}-bs-{}-lr-{}'.format(args.loader, args.batch_size, args.loss_rate))
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
        tmp_save_path='tmp_save',
    )
    runner.train_multi_epoch(args.epochs)
    reporting.save_all()


if __name__ == "__main__":
    main()
