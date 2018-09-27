'''Load image/labels/boxes from an annotation file.

The list file is like:

    img.jpg xmin ymin xmax ymax label xmin ymin xmax ymax label ...
'''
from __future__ import print_function

from glob import glob
import os
import sys
import random

from imblearn.under_sampling import RandomUnderSampler
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from retinanet.encoder import DataEncoder
from retinanet.transform import resize, random_flip, random_crop, center_crop


class ListDataset(data.Dataset):
    def __init__(self, root, list_file, train, transform, input_size, val=False, only_uni_or_bilateral=False, undersample=None, preprocessed=False):
        '''
        Args:
          root: (str) ditectory to images.
          list_file: (str) path to index file.
          train: (boolean) train or test.
          transform: ([transforms]) image transforms.
          input_size: (int) model input size.
          val: (bool) is this a validation dataset?
          only_uni_or_bilateral: (bool) only get unilateral or bilaterial pneumonia
          undersample: (float) undersample normal obs to a ratio of abnormal. Set as None if undesired for use
          preprocessed: (bool) is the dataset preprocessed in some way where the files are png?
        '''
        self.root = root
        self.train = train
        self.transform = transform
        self.input_size = input_size
        self.val = val

        self.fnames = []
        self.boxes = []
        self.labels = []
        self.preprocessed = preprocessed

        self.encoder = DataEncoder()

        # this is a modification for RSNA
        if not self.train and not self.val:
            self.fnames = [os.path.basename(f) for f in glob(os.path.join(root, '*.dcm'))]
            self.num_samples = len(self.fnames)
            return

        # open the indexer and get all images in the set
        with open(list_file) as f:
            lines = f.readlines()

        self.num_samples = 0
        for line in lines:
            splited = line.strip().split(',')
            # figure out how many bounding boxes are in an image
            num_boxes = (len(splited) - 1) // 5
            if only_uni_or_bilateral and num_boxes > 2:
                continue
            self.num_samples += 1
            # add filename
            self.fnames.append(splited[0])
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

        if undersample is not None:
            rus_labels = np.array([0 if 0 not in label.numpy() else 1 for label in self.labels])
            minority_len = len(rus_labels[rus_labels == 1])
            rus_idx = np.arange(0, len(self.labels)).reshape(-1, 1)
            # add replacement or no?
            rus = RandomUnderSampler(ratio={0: int(minority_len*undersample), 1: minority_len}, random_state=1)
            x, _ = rus.fit_sample(rus_idx, rus_labels)
            # add index to x so we can keep track of what gets shuffled
            x = x[:,0]
            np.random.shuffle(x)
            self.num_samples = len(x)
            self.boxes = [self.boxes[i] for i in x]
            self.labels = [self.labels[i] for i in x]
            self.fnames = [self.fnames[i] for i in x]

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
        if self.preprocessed:
            fname = fname.replace('.dcm', '.png')
            img = Image.open(os.path.join(self.root, fname))
            # resize back to normal dicom standard
            img = transforms.Resize((1024, 1024))(img)
        else:
            img = pydicom.read_file(os.path.join(self.root, fname))
            img = img.pixel_array
            if len(img.shape) != 3 or image.shape[2] != 3:
                img = np.stack((img,) * 3, -1)
            img = Image.fromarray(img)

        if self.train or self.val:
            boxes = self.boxes[idx].clone()
            labels = self.labels[idx]

        size = self.input_size
        # Data augmentation.
        if self.train:
            #img, boxes = random_flip(img, boxes)
            #img, boxes = random_crop(img, boxes)
            img, boxes = resize(img, boxes, (size,size))
            img = self.transform(img)
            return img, boxes, labels
        elif not self.train and not self.val:
            img = self.transform(img)
            return img, fname.replace('.dcm', '')
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
    from encoder import DataEncoder
    cxr14_norms = [[0.5059, 0.5059, 0.5059], [0.0893, 0.0893, 0.0893]]
    imagenet_norms = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]

    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(cxr14_norms[0], cxr14_norms[1])
    ])
    dataset = ListDataset(
        root='/fastdata/rsna-pneumonia/train_segmented',
        list_file='rsna-train.csv',
        train=True,
        transform=transform,
        input_size=224,
        preprocessed=True
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=dataset.collate_fn)

    for images, loc_targets, cls_targets in dataloader:
        print(images.size())
        print(loc_targets.size())
        print(cls_targets.size())
        print((cls_targets > 0).any())
        grid = torchvision.utils.make_grid(images, 1)
        torchvision.utils.save_image(grid, 'a.jpg')
        break

#test()
