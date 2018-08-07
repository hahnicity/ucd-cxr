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


def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BILINEAR), \
           boxes*torch.Tensor([sw,sh,sw,sh])


def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.

    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x,y,x,y])
    boxes[:,0::2].clamp_(min=0, max=w-1)
    boxes[:,1::2].clamp_(min=0, max=h-1)
    return img, boxes


def center_crop(img, boxes, size):
    '''Crops the given PIL Image at the center.

    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size (tuple): desired output size of (w,h).

    Returns:
      img: (PIL.Image) center cropped image.
      boxes: (tensor) center cropped boxes.
    '''
    w, h = img.size
    ow, oh = size
    i = int(round((h - oh) / 2.))
    j = int(round((w - ow) / 2.))
    img = img.crop((j, i, j+ow, i+oh))
    boxes -= torch.Tensor([j,i,j,i])
    boxes[:,0::2].clamp_(min=0, max=ow-1)
    boxes[:,1::2].clamp_(min=0, max=oh-1)
    return img, boxes


def random_flip(img, boxes):
    '''Randomly flip the given PIL Image.

    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].

    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:,0] = xmin
        boxes[:,2] = xmax
    return img, boxes


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
