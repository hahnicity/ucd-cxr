import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.init import kaiming_init, xavier_init
from cxrlib.models.resnet_grayscale import resnet50
from cxrlib.read_data import get_openi_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel


class RunOpenIModel(RunModel):
    def post_validation_actions(self):
        epoch_loss = self.reporting.get_meter('validation_epoch_loss').values
        mean_loss = epoch_loss.mean()
        self.lr_scheduler.step(mean_loss)


def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('images_path')
    parser.add_argument('--train-labels-path', default='/fastdata/openi/train_list.processed')
    parser.add_argument('--val-labels-path', default='/fastdata/openi/val_list.processed')
    parser.add_argument('--test-labels-path', default='/fastdata/openi/test_list.processed')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    parser.add_argument('--num-workers', default=multiprocessing.cpu_count(), type=int)
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--weight-init', choices=['xavier', 'kaiming'], default='xavier')
    # model hyperparameters
    args = parser.parse_args()

    model = resnet50(num_classes=20)
    if args.weight_init == 'kaiming':
        model.apply(kaiming_init)
    elif args.weight_init == 'xavier':
        model.apply(xavier_init)
    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False
    train_loader, valid_loader, test_loader = get_openi_loaders(
        args.images_path,
        args.train_labels_path,
        args.val_labels_path,
        args.test_labels_path,
        args.batch_size,
        convert_to='LA',
        is_preprocessed=is_preprocessed,
        num_workers=args.num_workers
    )
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = cuda_wrapper(torch.nn.DataParallel(model))
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'model', False)
    runner = RunModel(
        args,
        model,
        train_loader,
        test_loader,
        optimizer,
        lr_scheduler,
        criterion,
        True if args.device == 'cuda' else False,
        reporting,
        validation_loader=valid_loader
    )
    runner.train_multi_epoch(args.epochs)
    del train_loader
    del valid_loader
    torch.cuda.empty_cache()
    runner.generic_test_epoch()
    reporting.save_all('openi-resnet50-grayscale')


if __name__ == "__main__":
    main()
