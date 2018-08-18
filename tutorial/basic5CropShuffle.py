import argparse
import multiprocessing
import os

import torch
from torchvision import transforms

from cxrlib.models import guan_resnet_ag
from cxrlib.read_data import get_five_crop_loaders
from cxrlib.results import Reporting
from cxrlib.run import RunModel
import torch.utils.model_zoo as model_zoo

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class GuanAGRun(RunModel):
    def post_epoch_actions(self):
        self.lr_scheduler.step()

def main():
    parser = argparse.ArgumentParser()
    # run related options
    parser.add_argument('--images-path', default='/media/minh/UStorage/chestxray14/images/')
    parser.add_argument('--labels-path', default='/media/minh/UStorage/chestxray14/labels/')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--results-path', default=os.path.join(os.path.dirname(__file__), 'results'))
    parser.add_argument('--print-progress', action='store_true')
    # training options
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    # model hyperparameters
    args = parser.parse_args()

    model = guan_resnet_ag.GuanResNet50_AG().cuda()
    model.load_my_state_dict(model_zoo.load_url(model_urls['resnet50']))
    train_loader, test_loader = get_five_crop_loaders(args.images_path,args.labels_path,args.batch_size)
    cuda_wrapper = lambda x: x.cuda() if args.device == 'cuda' else x
    model = cuda_wrapper(torch.nn.DataParallel(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)
    criterion = torch.nn.BCELoss()
    reporting = Reporting(args.results_path,file_suffix="Testing")
    reporting.register(model, 'model', False)
    runner = GuanAGRun(
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
    reporting.save_all('guan-AG-pretrained-{}'.format(args.pretrained))

if __name__ == "__main__":
    main()

