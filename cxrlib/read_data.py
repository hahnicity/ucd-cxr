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

from cxrlib.constants import CLASS_NAMES, CXR14_LA_NORM, CXR14_RGB_NORM, IMAGENET_NORM
from cxrlib import transforms as cxr_transforms


class BBoxChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, transform=None, convert_to='RGB'):
        """
        Get Dataset for bounding box images

        :param data_dir: path to image directory.
        :param image_list_file: path to file where filenames are matched with bounding box coords
        :param is_preprocessed: if the data directory contains preprocessed information. If so do no transformations
        :param transform: optional transform to be applied on a sample.
        :param convert_to: convert the image to rgb (RGB) or grayscale (LA)
        """
        self.fnames = []
        self.boxes = []
        self.labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                split = [i for i in line.strip().split() if i != 'nan']
                # -15 because of the 14 GT annos plus 1 img filename
                num_boxes = (len(split) - 15) // 5
                self.fnames.append(os.path.join(data_dir, split[0]))
                label = []
                box = []
                for i in range(num_boxes):
                    xmin = float(split[15+5*i])
                    ymin = float(split[16+5*i])
                    xmax = float(split[17+5*i])
                    ymax = float(split[18+5*i])
                    label.append(float(split[19+5*i]))
                    box.append([xmin, ymin, xmax, ymax])
                self.boxes.append(box)
                self.labels.append(label)

        self.transform = transform
        self.convert_to = convert_to
        self.is_train = 'train' in image_list_file

    def __getitem__(self, index):
        """
        :param index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.fnames[index]
        labels = self.labels[index]
        boxes = self.boxes[index]

        image = Image.open(image_name).convert(self.convert_to)

        if self.is_train:
            img, boxes = cxr_transforms.random_flip(img, boxes)
            img, boxes = cxr_transforms.random_crop(img, boxes)
            img, boxes = cxr_transforms.resize(img, boxes, (224, 224))
        else:
            img, boxes = cxr_transforms.resize(img, boxes, (256, 256))
            img, boxes = cxr_transforms.center_crop(img, boxes, (224, 224))
        image = self.transform(image)

        # We're using multi-crop here
        if image.ndimension() == 4:
            # XXX figure out boxes on multi-crop
            raise NotImplementedError('multi-crop is not implemented yet for bounding boxes')
            ncrops = image.size(0)
            labels = torch.FloatTensor([label])

            for _ in range(ncrops-1):
                labels = torch.cat((labels, torch.FloatTensor([label])), 0)
            return image, labels
        else:
            return image, torch.FloatTensor(boxes), torch.FloatTensor(labels)

    def __len__(self):
        return len(self.fnames)


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, is_preprocessed=False, transform=None, convert_to='RGB', focus_on=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            is_preprocessed: if the data directory contains preprocessed information. If so do no transformations
            transform: optional transform to be applied on a sample.
            convert_to: convert the image to rgb (RGB) or grayscale (LA)
            focus_on: focus on a specific class
        """
        image_names = []
        labels = []
        if focus_on and focus_on not in CLASS_NAMES:
            raise Exception('focus_on var must be set to be one of: {}'.format(CLASS_NAMES))
        if focus_on:
            focus_idx = CLASS_NAMES.index(focus_on)

        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                if focus_on:
                    label = [int(items[1:][focus_idx])]
                else:
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
        else:
            image = torch.load(image_name)

        # We're using multi-crop here
        if image.ndimension() == 4:
            ncrops = image.size(0)
            labels = torch.FloatTensor([label])
            for _ in range(ncrops-1):
                labels = torch.cat((labels, torch.FloatTensor([label])), 0)
            return image, labels
        # We're not using multi-crop here
        else:
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


def _get_transforms(trans_type, norms, convert_to):
    """
    :param trans_type: Can be guan, five_crop, openi, baltruschat, or bbox
    :param norms: Can be cxr14 or imagenet
    :param convert_to: Can be RGB or LA
    """
    if norms == 'cxr14' and convert_to == 'RGB':
        norms = CXR14_RGB_NORM
    elif norms == 'cxr14' and convert_to == 'LA':
        norms = CXR14_LA_NORM
    elif norms == 'imagenet':
        norms = IMAGENET_NORM

    if convert_to == 'RGB':
        if trans_type == 'five_crop':
            train_transforms, test_transforms = cxr_transforms.five_crop_rgb_transforms(norms)
        elif trans_type == 'guan':
            train_transforms, test_transforms = cxr_transforms.guan_rgb_transforms(norms)
        elif trans_type == 'baltruschat':
            train_transforms, test_transforms = cxr_transforms.baltruschat_rgb_transforms(norms)
        elif trans_type == 'openi':
            train_transforms, test_transforms = cxr_transforms.openi_rgb_transforms(norms)
        elif trans_type == 'bbox':
            return cxr_transforms.get_bbox_rgb_transforms(norms)

    elif convert_to == 'LA':
        if trans_type == 'five_crop':
            train_transforms, test_transforms = cxr_transforms.five_crop_grayscale_transforms(norms)
        elif trans_type == 'guan':
            train_transforms, test_transforms = cxr_transforms.guan_grayscale_transforms(norms)
        elif trans_type == 'baltruschat':
            train_transforms, test_transforms = cxr_transforms.baltruschat_grayscale_transforms(norms)
        elif trans_type == 'openi':
            train_transforms, test_transforms = cxr_transforms.openi_grayscale_transforms(norms)
        elif trans_type == 'bbox':
            return cxr_transforms.get_bbox_grayscale_transforms(norms)

    return train_transforms, test_transforms


def _get_loaders(images_path,
                 labels_path,
                 batch_size,
                 num_workers,
                 convert_to,
                 norms,
                 train_transforms,
                 test_transforms,
                 is_preprocessed,
                 get_validation_set,
                 focus_on):
    """
    Get data loaders for CXR14 or OpenI

    :param images_path: path to directory where all images are located
    :param train_labels_path: full path to train labels
    :param valid_labels_path: full path to validation labels
    :param test_labels_path: full path to test labels
    :param batch_size: size of mini-batches for train and test sets
    :param num_workers: number of cpu workers to use when loading data
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    :param train_transforms: Compose object with train transformations
    :param test_transforms: Compose object with test transformations
    :param is_preprocessed: is the dataset preprocessed? Does it need transforms?
    :param get_validation_set: True/False if we want a validation set
    :param focus_on: XXX
    """

    if get_validation_set:
        train_labels_path = os.path.join(labels_path, 'train_list.processed')
    else:
        train_labels_path = os.path.join(labels_path, "train_val_list.processed")

    train_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=train_labels_path,
        transform=train_transforms,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
        focus_on=focus_on,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=os.path.join(labels_path, "test_list.processed"),
        transform=test_transforms,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
        focus_on=focus_on,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    if get_validation_set:
        valid_dataset = ChestXrayDataSet(
            data_dir=images_path,
            image_list_file=os.path.join(labels_path, 'val_list.processed'),
            transform=train_transforms,
            convert_to=convert_to,
            is_preprocessed=is_preprocessed,
            focus_on=focus_on,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset, batch_size=batch_size,
            shuffle=True, num_workers=num_workers, pin_memory=True
        )
        return train_loader, valid_loader, test_loader
    else:
        return train_loader, test_loader


def get_bbox_loaders(images_path,
                     labels_dir,
                     batch_size,
                     bbox_batch_size,
                     convert_to,
                     norms='cxr14'):
    """
    Get data loaders for BBox CXR14

    :param images_path: path to directory where all images are located
    :param labels_dir: dir where dataset resides
    :param batch_size: size of mini-batches for train and test sets
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    """

    train_labels_nobbox_path = os.path.join(labels_dir, 'bbox_train_nobbox.processed')
    train_labels_bbox_path = os.path.join(labels_dir, 'bbox_train_withbbox.processed')
    test_labels_nobbox_path = os.path.join(labels_dir, 'bbox_test_nobbox.processed')
    test_labels_bbox_path = os.path.join(labels_dir, 'bbox_test_withbbox.processed')

    train_transforms, train_bbox_transforms, test_transforms, test_bbox_transforms = _get_transforms('bbox', norms, convert_to)
    num_workers = multiprocessing.cpu_count()

    train_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=train_labels_nobbox_path,
        transform=train_transforms,
        convert_to=convert_to,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    train_bbox_dataset = BBoxChestXrayDataSet(
        data_dir=images_path,
        image_list_file=train_labels_bbox_path,
        transform=train_bbox_transforms,
        convert_to=convert_to,
    )
    train_bbox_loader = torch.utils.data.DataLoader(
        dataset=train_bbox_dataset, batch_size=bbox_batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=test_labels_nobbox_path,
        transform=test_transforms,
        convert_to=convert_to,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_bbox_dataset = BBoxChestXrayDataSet(
        data_dir=images_path,
        image_list_file=test_labels_bbox_path,
        transform=test_bbox_transforms,
        convert_to=convert_to,
    )
    test_bbox_loader = torch.utils.data.DataLoader(
        dataset=test_bbox_dataset, batch_size=bbox_batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    return train_loader, train_bbox_loader, test_loader, test_bbox_loader


def get_loaders(images_path, labels_path, batch_size, num_workers=multiprocessing.cpu_count(), convert_to='RGB', norms='imagenet', is_preprocessed=False, get_validation_set=False, transform_type='guan', focus_on=None):
    """
    Get data loaders for either CXR14 or OpenI

    :param images_path: path to directory where all images are located
    :param labels_path: path to directory where all labels are located
    :param batch_size: size of mini-batches for train and test sets
    :param num_workers: number of cpu workers to use when loading data
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    :param is_preprocessed: is the dataset preprocessed? Does it need transforms?
    :param get_validation_set: Return the validation loader or no?
    :param transform_type: The type of transforms we want to perform on our image choices: openi, guan, five_crop, baltruschat
    :param focus_on: XXX
    """
    train_transforms, test_transforms = _get_transforms(transform_type, norms, convert_to)
    return _get_loaders(images_path, labels_path, batch_size, num_workers, convert_to, norms, train_transforms, test_transforms, is_preprocessed, get_validation_set, focus_on)


def get_openi_loaders(images_path, train_labels_path, valid_labels_path, test_labels_path, batch_size, num_workers=multiprocessing.cpu_count(), convert_to='RGB', norms='cxr14', is_preprocessed=False):
    """
    Get data loaders for OpenI dataset. Since OpenI is significantly smaller than
    CXR14 we perform some transforms to boost the amount of data that we have.

    DEPRECATED, DO NOT USE, USE get_loaders instead

    :param images_path: path to directory where all images are located
    :param train_labels_path: full path to train labels
    :param valid_labels_path: full path to validation labels
    :param test_labels_path: full path to test labels
    :param batch_size: size of mini-batches for train and test sets
    :param num_workers: number of cpu workers to use when loading data
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    :param is_preprocessed: is the dataset preprocessed? Does it need transforms?
    """
    train_transforms, test_transforms = _get_transforms('openi', norms, convert_to)
    train_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=train_labels_path,
        transform=train_transforms,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    valid_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=valid_labels_path,
        transform=train_transforms,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    # XXX in future allow option for no test dataset so that we can do pure
    # pretraining
    test_dataset = ChestXrayDataSet(
        data_dir=images_path,
        image_list_file=test_labels_path,
        transform=test_transforms,
        convert_to=convert_to,
        is_preprocessed=is_preprocessed,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    return train_loader, valid_loader, test_loader


def get_guan_loaders(images_path, labels_path, batch_size, num_workers=multiprocessing.cpu_count(), convert_to='RGB', norms='cxr14', is_preprocessed=False, get_validation_set=False):
    """
    Get data loaders for Guan method. For initial prototyping these data loaders can
    be useful. However, there are still improvements that can be made and it should not
    be used long-term

    DEPRECATED, DO NOT USE, USE get_loaders instead

    :param images_path: path to directory where all images are located
    :param labels_path: path to directory where all labels are located
    :param batch_size: size of mini-batches for train and test sets
    :param num_workers: number of cpu workers to use when loading data
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    :param is_preprocessed: is the dataset preprocessed? Does it need transforms?
    :param get_validation_set: Return the validation loader or no?
    :param transform_type: The type of transforms we want to perform on our image choices: openi, guan, five_crop, baltruschat
    """
    train_transforms, test_transforms = _get_transforms('guan', norms, convert_to)
    return _get_loaders(images_path, labels_path, batch_size, num_workers, convert_to, norms, train_transforms, test_transforms, is_preprocessed, get_validation_set)


def get_five_crop_loaders(images_path, labels_path, batch_size, num_workers=multiprocessing.cpu_count(), convert_to='RGB', norms='cxr14', is_preprocessed=False, get_validation_set=False):
    """
    Get data loaders for Five Crop method.

    DEPRECATED, DO NOT USE, USE get_loaders instead

    :param images_path: path to directory where all images are located
    :param labels_path: path to directory where all labels are located
    :param batch_size: size of mini-batches for train and test sets
    :param num_workers: number of cpu workers to use when loading data
    :param convert_to: convert images to RGB or LA (for grayscale)
    :param norms: the dataset normalization standard we want to use. Accepts cxr14 and imagenet
    :param is_preprocessed: is the dataset preprocessed? Does it need transforms?
    :param get_validation_set: Return the validation loader or no?
    """
    train_transforms, test_transforms = _get_transforms('five_crop', norms, convert_to)
    return _get_loaders(images_path, labels_path, batch_size, num_workers, convert_to, norms, train_transforms, test_transforms, is_preprocessed, get_validation_set)
