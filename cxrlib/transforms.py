from PIL.ImageOps import grayscale as to_grayscale
import torch
from torchvision import transforms


class ToGrayscaleTensor(object):
    """
    Converts a CXR image to a tensor and grayscale
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        image = to_grayscale(image)
        image = self.to_tensor(image)
        # If we're converting to grayscale and there's a fourth channel
        # then error out.
        if image.size(0) == 2 and (image[1] == 1).all():
            raise Exception('Tried to convert image to grayscale but more than 1 channel exists!')
        return image


def guan_rgb_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def guan_grayscale_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        ToGrayscaleTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToGrayscaleTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def openi_grayscale_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
		transforms.Lambda(
			lambda crops: torch.stack([ToGrayscaleTensor()(crop) for crop in crops])
		),
		transforms.Lambda(
			lambda crops: torch.stack([normalize(crop) for crop in crops])
		),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToGrayscaleTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def openi_rgb_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.TenCrop(224),
		transforms.Lambda(
			lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])
		),
		transforms.Lambda(
			lambda crops: torch.stack([normalize(crop) for crop in crops])
		),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def five_crop_rgb_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
		transforms.Lambda(
			lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])
		),
		transforms.Lambda(
			lambda crops: torch.stack([normalize(crop) for crop in crops])
		),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def five_crop_grayscale_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
		transforms.Lambda(
			lambda crops: torch.stack([ToGrayscaleTensor()(crop) for crop in crops])
		),
		transforms.Lambda(
			lambda crops: torch.stack([normalize(crop) for crop in crops])
		),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToGrayscaleTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def baltruschat_grayscale_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        # aspect ratio needs no change (3/4 to 4/3)
        transforms.RandomResizedCrop(224, scale=(.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        ToGrayscaleTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        ToGrayscaleTensor(),
        normalize,
    ])
    return train_transforms, test_transforms


def baltruschat_rgb_transforms(norms):
    normalize = transforms.Normalize(*norms)
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        # aspect ratio needs no change (3/4 to 4/3)
        transforms.RandomResizedCrop(224, scale=(.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(7),
        transforms.ToTensor(),
        normalize,
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transforms, test_transforms
