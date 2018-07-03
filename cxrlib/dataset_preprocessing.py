"""
dataset_preprocessing
~~~~~~~~~~~~~~~~~~~~~

Compute tensor operations on images and save them as a preprocessed
file so that we don't necessarily have to do so during training.

This is good if we want to perform slightly less processing when prototyping
new methods, but should not be utilized when we are actually evaluating
how well a model performs in reality. The only thing we can possibly do
when performing a true evaluation is to let the code perform all transforms
itself so we can keep some non-determinism.
"""
import argparse
from glob import glob
import os

from PIL import Image
import torch
from torchvision import transforms

from cxrlib.constants import CXR14_LA_NORM, CXR14_RGB_NORM


def guan_transforms(norm):
    normalize = transforms.Normalize(*norm)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def process_list(list_path, transforms, args):
    with open(list_path, 'r') as f:
        for line in f:
            items = line.split()
            image_name = items[0]
            image_path = os.path.join(args.input_dir, image_name)
            img = Image.open(image_path).convert(args.convert_to)
            img = transforms(img)
            # If we're converting to grayscale and there's a fourth channel
            # that we don't want, then remove it
            if img.size(0) == 2 and args.convert_to == 'LA':
                img = img[0].view(1, img.size(1), img.size(2))
            filepath = os.path.join(args.output_dir, os.path.basename(image_path))
            torch.save(img, filepath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    parser.add_argument('--convert-to', choices=['LA', 'RGB'], default='LA')
    parser.add_argument('--labels-path', default='/fastdata/chestxray14/labels')
    args = parser.parse_args()

    if args.input_dir == args.output_dir:
        raise Exception("you cannot have your input and output dirs be the same place!")

    train_list = os.path.join(args.labels_path, 'train_val_list.processed')
    test_list = os.path.join(args.labels_path, 'test_list.processed')
    if args.convert_to == 'LA':
        norm = CXR14_LA_NORM
    elif args.convert_to == 'RGB':
        norm = CXR14_RGB_NORM

    # XXX use guan transforms for now, but later we may want to preprocess 5-10 crop
    train_transforms, test_transforms = guan_transforms(norm)
    process_list(train_list, train_transforms, args)
    process_list(test_list, test_transforms, args)


if __name__ == "__main__":
    main()
