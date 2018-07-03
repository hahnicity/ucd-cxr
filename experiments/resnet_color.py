import argparse
import multiprocessing
import os

import torch
from torchvision import transforms
from torchvision.models.resnet import resnet50

from cxrlib.init import kaiming_init, xavier_init
from cxrlib.read_data import get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel


class Resnet50GrayscaleRun(RunModel):
    def post_epoch_actions(self):
        self.lr_scheduler.step()


def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('images_path')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--weight-init', choices=['xavier', 'kaiming'], default='xavier')
    # model hyperparameters
    args = parser.parse_args()

    model = resnet50(pretrained=args.pretrained, num_classes=14)
    if args.weight_init == 'kaiming' and not args.pretrained:
        model.apply(kaiming_init)
    elif args.weight_init == 'xavier' and not args.pretrained:
        model.apply(xavier_init)
    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False
    train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=is_preprocessed)
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = cuda_wrapper(torch.nn.DataParallel(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'model', False)
    runner = Resnet50GrayscaleRun(
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
    reporting.save_all('resnet50-color')


if __name__ == "__main__":
    main()
