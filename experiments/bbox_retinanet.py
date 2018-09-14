import argparse
import multiprocessing
import os

import torch

from cxrlib.models.retinanet import RetinaNet
from cxrlib.read_data import get_bbox_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModelWithAUCAndValLR


def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('--images_path', default='/fastdata/chestxray14/images/')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    # training options
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--bbox-batch-size', default=4, type=int)
    parser.add_argument('-r', '--run-test-after-epoch', type=int, help='run testing auc calculations after epoch N', default=0)
    # model hyperparameters
    parser.add_argument('-lr', '--learning-rate', type=float, default=.001)
    parser.add_argument('--nesterov', action='store_true')
    args = parser.parse_args()

    train_loader, train_bbox_loader, test_loader, test_bbox_loader = get_bbox_loaders(args.images_path, args.labels_path, args.batch_size, args.bbox_batch_size, 'RGB')
    model = RetinaNet(output_locs=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=1e-5)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, mode='min')
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path, 'retinanet-bs-{}-lr-{}'.format(args.batch_size, args.learning_rate))
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
        tmp_save_path='tmp_save',
        train_bbox_loader=train_bbox_loader,
        test_bbox_loader=test_bbox_loader,
    )
    runner.generic_train_epoch(1)


if __name__ == "__main__":
    main()
