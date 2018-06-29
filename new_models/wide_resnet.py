import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.models import WideResNet
from cxrlib.read_data import ChestXrayDataSet
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
    args = parser.parse_args()

    model = WideResNet(40, 2, .3, 14)
    # XXX ensure model can work on cpu
    model = torch.nn.DataParallel(model).cuda()
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ChestXrayDataSet(
        data_dir=args.images_path,
        image_list_file=os.path.join(args.labels_path, "train_val_list.processed"),
        transform=transformations
    )
    # XXX ensure these loaders can work on cpu too
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
    )
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = ChestXrayDataSet(
        data_dir=args.images_path,
        image_list_file=os.path.join(args.labels_path, "test_list.processed"),
        transform=transformations
    )
    # XXX ensure these loaders can work on cpu too
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=multiprocessing.cpu_count(), pin_memory=True
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=.2, momentum=.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'wide-resnet-22-2', False)
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
    # XXX make sure that we can work with cpu
    del train_loader
    torch.cuda.empty_cache()
    runner.generic_test_epoch()
    reporting.save_all('wide_resnet')

if __name__ == "__main__":
    main()
