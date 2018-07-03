# encoding: utf-8

"""
Read images and corresponding labels.
"""
import multiprocessing
import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from cxrlib.constants import CXR14_LA_NORM, CXR14_RGB_NORM, IMAGENET_NORM


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, is_preprocessed=False, transform=None, convert_to='RGB'):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            is_preprocessed: if the data directory contains preprocessed information. If so do no transformations
            transform: optional transform to be applied on a sample.
            convert_to: convert the image to rgb (RGB) or grayscale (LA)
        """
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.convert_to = convert_to
        self.is_preprocessed = is_preprocessed

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        label = self.labels[index]
        if self.transform and not self.is_preprocessed:
            image = Image.open(image_name).convert(self.convert_to)
            image = self.transform(image)
            # If we're converting to grayscale and there's a fourth channel
            # that we don't want, then remove it
            if image.size(0) == 2 and self.convert_to == 'LA':
                image = image[0].view(1, image.size(1), image.size(2))
        else:
            image = torch.load(image_name)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)


class RandomDataset(Dataset):
    def __init__(self, transform=None, var_max=5, n_items=20):
        """
        Generates random data for example purposes. Target is a single variable
        """
        self.data = [
            (torch.rand(3, 224, 224), torch.FloatTensor([np.random.randint(var_max+1)]))
            for _ in range(n_items)
        ]
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.data)


def get_guan_loaders(images_path, labels_path, batch_size, num_workers=multiprocessing.cpu_count(), convert_to='RGB', norms='cxr14', is_preprocessed=False):
    """
    Get data loaders for Guan method. For initial prototyping these data loaders can
    be useful. However, there are still improvements that can be made and it should not
    be used long-term

    :param images_path: path to directory where all images are located
    :param labels_path: path to directory where all labels are located
    :param batch_size: size of mini-batches for train and test sets
    :param num_workers: number of cpu workers to use when loading data
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    :param is_preprocessed: is the dataset preprocessed? Does it need transforms?
    """
    if norms == 'cxr14' and convert_to == 'RGB':
        norms = CXR14_RGB_NORM
    elif norms == 'cxr14' and convert_to == 'LA':
        norms = CXR14_LA_NORM
    elif norms == 'imagenet':
        norms = IMAGENET_NORM
    normalize = transforms.Normalize(*norms)
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=os.path.join(labels_path, "train_val_list.processed"),
        transform=transformations,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=os.path.join(labels_path, "test_list.processed"),
        transform=transformations,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader
