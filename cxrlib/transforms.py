import torch
from torchvision import transforms


class ToGrayscaleTensor(object):
    """
    Converts a CXR image to a tensor and grayscale at once
    """
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image):
        image = image.convert('LA')
        image = self.to_tensor(image)
        # If we're converting to grayscale and there's a fourth channel
        # that we don't want, then remove it
        if image.size(0) == 2 and (image[1] == 1).all():
            image = image[0].view(1, image.size(1), image.size(2))
        # I haven't seen this happen but there's a first time for everything
        elif image.size(0) == 2 and not (image[1] == 1).all():
            raise Exception('Tried to convert image to grayscale but non-1 information exists in 4 dimension')
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
			lambda crops: torch.stack([transforms.ToGrayscaleTensor()(crop) for crop in crops])
		),
		transforms.Lambda(
			lambda crops: torch.stack([normalize(crop) for crop in crops])
		),
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToGrayscaleTensor(),
        normalize,
    ])
