'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

from glob import glob
import os
import sys
import random

import numpy as np
import pydicom
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from encoder import DataEncoder
from transform import resize, random_flip, random_crop, center_crop


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size, val=False):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
          val: (bool) is this a validation dataset?
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.val = val

        self.fnames = []
        self.boxes = []
        self.labels = []

        self.encoder = DataEncoder()

        # this is a modification for RSNA
        if not self.train and not self.val:
            self.fnames = [os.path.basename(f) for f in glob(os.path.join(root, '*.dcm'))]
            self.num_samples = len(self.fnames)
            return

        # open the indexer and get all images in the set
        with open(list_file) as f:
            lines = f.readlines()
            self.num_samples = len(lines)

        for line in lines:
            splited = line.strip().split(',')
            # add filename
            self.fnames.append(splited[0])
            # figure out how many bounding boxes are in an image
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            # get all stats for each box. Each box should have 4 position identifiers
            # and it is followed by a class identifier.
            for i in range(num_boxes):
                xmin = splited[1+5*i]
                ymin = splited[2+5*i]
                xmax = splited[3+5*i]
                ymax = splited[4+5*i]
                c = splited[5+5*i]
                box.append([float(xmin),float(ymin),float(xmax),float(ymax)])
                label.append(int(c))
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          loc_targets: (tensor) location targets.
          cls_targets: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx]
        # This is a code modification for the RSNA dataset
        img = pydicom.read_file(os.path.join(self.root, fname))
        img = img.pixel_array
        if len(img.shape) != 3 or image.shape[2] != 3:
            img = np.stack((img,) * 3, -1)
        img = Image.fromarray(img)
        #img = Image.open(os.path.join(self.root, fname))
        #if img.mode != 'RGB':
        #    img = img.convert('RGB')
        if self.train or self.val:
            boxes = self.boxes[idx].clone()
            labels = self.labels[idx]
            size = self.input_size

        # Data augmentation.
        if self.train:
            img, boxes = random_flip(img, boxes)
            img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
            img = self.transform(img)
            return img, boxes, labels
        elif not self.train and not self.val:
            img = self.transform(img)
            return img
        else:
            img, boxes = resize(img, boxes, size)
            img, boxes = center_crop(img, boxes, (size,size))
            img = self.transform(img)
            return img, boxes, labels

    def collate_fn(self, batch):
        '''Pad images and encode targets.

        As for images are of different sizes, we need to pad them to the same size.

        Args:
          batch: (list) of images, cls_targets, loc_targets.

        Returns:
          padded images, stacked cls_targets, stacked loc_targets.
        '''
        imgs = [x[0] for x in batch]
        boxes = [x[1] for x in batch]
        labels = [x[2] for x in batch]

        h = w = self.input_size
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 3, h, w)

        loc_targets = []
        cls_targets = []
        for i in range(num_imgs):
            inputs[i] = imgs[i]
            loc_target, cls_target = self.encoder.encode(boxes[i], labels[i], input_size=(w,h))
            loc_targets.append(loc_target)
            cls_targets.append(cls_target)
        return inputs, torch.stack(loc_targets), torch.stack(cls_targets)

    def __len__(self):
        return self.num_samples


def test():
    import torchvision

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    dataset = ListDataset(root='/mnt/hgfs/D/download/PASCAL_VOC/voc_all_images',
                          list_file='./data/voc12_train.txt', train=True, transform=transform, input_size=600)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

# test()
