import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.init import kaiming_init, xavier_init
from cxrlib.models.shake_shake import Network
from cxrlib.read_data import get_guan_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel


class ShakeShakeRun(RunModel):
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
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--weight-init', choices=['xavier', 'kaiming'], default='xavier')
    # model hyperparameters
    parser.add_argument('--depth', default=26, type=int)
    parser.add_argument('--base-channels', default=16, type=int)
    parser.add_argument('--no-shake-forward', action='store_false')
    parser.add_argument('--no-shake-backward', action='store_false')
    parser.add_argument('--no-shake-image', action='store_false')
    args = parser.parse_args()

    config = {
        'input_shape': (args.batch_size, 1, 224, 224),
        'n_classes': 14,
        'base_channels': args.base_channels,
        'depth': args.depth,
        # These options all come out to be true unless they're set on the CLI
        'shake_forward': args.no_shake_forward,
        'shake_backward': args.no_shake_backward,
        'shake_image': args.no_shake_image,
    }
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = cuda_wrapper(Network(config))
    if args.weight_init == 'kaiming':
        model.apply(kaiming_init)
    elif args.weight_init == 'xavier':
        model.apply(xavier_init)
    if "preprocessed" in args.images_path:
        is_preprocessed = True
    else:
        is_preprocessed = False
    train_loader, test_loader = get_guan_loaders(args.images_path, args.labels_path, args.batch_size, is_preprocessed=True, convert_to='LA')
    if not args.debug:
        model = cuda_wrapper(torch.nn.DataParallel(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
    criterion = torch.nn.BCEWithLogitsLoss()
    reporting = Reporting(args.results_path)
    reporting.register(model, 'shake-shake-depth{}-sf-{}-sb-{}-si-{}'.format(args.depth, not args.no_shake_forward, not args.no_shake_backward, not args.no_shake_image), False)
    runner = ShakeShakeRun(
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
    torch.cuda.empty_cache()
    runner.generic_test_epoch()
    reporting.save_all('shake-shake')


if __name__ == "__main__":
    main()
