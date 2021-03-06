import argparse
import multiprocessing
import os

import torch
from torchvision import transforms
from torchvision.models.resnet import resnet101

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
    parser.add_argument('--load-openi-model', help='specify path to stored openi model')
    parser.add_argument('--no-validation', action='store_true')
    parser.add_argument('--loader', choices=['baltruschat', 'five_crop', 'guan'], default='guan')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('-r', '--run-test-after-epoch', type=int, help='run testing auc calculations after epoch N', default=0)
    # model hyperparameters
    parser.add_argument('-lr', '--loss-rate', type=float, default=.001)
    args = parser.parse_args()

    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    if not args.load_openi_model:
        model = resnet101(pretrained=True)
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

    train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed, transform_type=args.loader)
    valid_loader = None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.loss_rate, weight_decay=1e-4, eps=1e-8)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path, 'resnet101-loader-{}-bs-{}-lr-{}-adam'.format(args.loader, args.batch_size, args.loss_rate))
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
    )
    runner.train_multi_epoch(args.epochs)
    del train_loader
    del valid_loader
    torch.cuda.empty_cache()
    runner.generic_test_epoch()
    reporting.save_all()


if __name__ == "__main__":
    main()
