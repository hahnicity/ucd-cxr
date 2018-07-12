import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.loss import FocalLoss
from cxrlib.models.guan_resnet import GuanResNet50
from cxrlib.read_data import get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel


class GuanRun(RunModel):
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
    # training options
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    # model hyperparameters
    args = parser.parse_args()

    model = GuanResNet50(pretrained=args.pretrained)
    train_loader, valid_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, get_validation_set=True)
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = cuda_wrapper(torch.nn.DataParallel(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    criterion = FocalLoss(.5)
    reporting = Reporting(args.results_path)
    reporting.register(model, 'model', False)
    runner = GuanRun(
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
    reporting.save_all('guan-rgb-pretrained-{}-focal-loss'.format(args.pretrained))


if __name__ == "__main__":
    main()
