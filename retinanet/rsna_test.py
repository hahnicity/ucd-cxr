import argparse
import multiprocessing

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from encoder import DataEncoder
from datagen import ListDataset
from retinanet import RetinaNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file')
    parser.add_argument('test_image_dir')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint_file)
    net = RetinaNet(num_classes=1).cuda()
    net.load_state_dict(ckpt['net'])
    input_size = 224

    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    data = ListDataset(args.test_image_dir, None, False, test_transforms, input_size)
    loader = DataLoader(data, batch_size=1, num_workers=multiprocessing.cpu_count())
    encoder = DataEncoder()
    net.eval()
    with open('kaggle-submission.txt', 'w') as sub_file:
        # XXX write header
        for inputs in loader:
            inputs = Variable(inputs, volatile=True).cuda()
            loc_preds, cls_preds = net(inputs)
            boxes, labels = encoder.decode(loc_preds.data.detach().cpu().squeeze(), cls_preds.data.detach().cpu().squeeze(), (input_size, input_size))
            all_lab = set(labels.numpy().ravel())
            print(all_lab)
            print(boxes.size())


if __name__ == "__main__":
    main()
