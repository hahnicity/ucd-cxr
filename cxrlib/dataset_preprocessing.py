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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    if args.input_dir == args.output_dir:
        raise Exception("you cannot have your input and output dirs be the same place!")

    # These transforms can be modified in the future. But for now, just use the ones
    # that Guan 2018 used because they are reasonable and won't blow up storage space
    # like 10 fold will
    #
    # XXX also we really need to not use imagenet constants here
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
    methods = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    files = glob(os.path.join(args.input_dir, '*.png'))
    for file_ in files:
        img = Image.open(file_)
        img = methods(img)
        filepath = os.path.join(args.output_dir, os.path.basename(file_))
        torch.save(img, filepath)


if __name__ == "__main__":
    main()
