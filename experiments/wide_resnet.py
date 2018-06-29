import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.models.wide_resnet import WideResNet
from cxrlib.read_data import get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel


class WideResNetRun(RunModel):
    def post_epoch_actions(self):
        self.lr_scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images_path')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('-k', default=2, type=int, help='expansion factor')
    parser.add_argument('-d', '--depth', type=int, help='layer depth')
    parser.add_argument('-r', '--dropout-rate', type=float, default=.3)
    args = parser.parse_args()

    model = WideResNet(args.depth, args.k, args.dropout_rate, 14)
    train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size)
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = cuda_wrapper(torch.nn.DataParallel(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=.2, momentum=.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'wide-resnet-{}-{}-rate-'.format(args.depth, args.k, args.dropout_rate), False)
    runner = WideResNetRun(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        criterion,
        True if args.device == 'cuda' else False,
        reporting
    )
    runner.train_multi_epoch(args.epochs)
    del train_loader
    torch.cuda.empty_cache()
    runner.generic_test_epoch()
    reporting.save_all('wide_resnet')


if __name__ == "__main__":
    main()
