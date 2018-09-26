from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

class visualize(object):
    def __init__(self, sample):
        self.sample = sample

    def ImageWithGround(self, idx, left=False, right=False, save=False):
        background = np.asarray(self.sample[idx]['image'])
        plt.imshow(background[0].astype('float'),cmap='gray')
        filter = np.asarray(self.sample[idx]['label'],dtype='uint8')
        if left:
            leftfilter = filter == 1
            leftfilter.astype('uint8')
            leftfilter = leftfilter*255
            zerolayer = np.zeros((leftfilter.shape[0],leftfilter.shape[1]))
            leftforeground = np.stack((zerolayer,zerolayer,leftfilter),axis=-1)
            plt.imshow(leftforeground.astype('uint8'),alpha=0.3)
        if right:
            rightfilter = filter == 2
            rightfilter.astype('uint8')
            rightfilter = rightfilter*255
            zerolayer = np.zeros((rightfilter.shape[0],rightfilter.shape[1]))
            rightforeground = np.stack((rightfilter,zerolayer,zerolayer),axis=-1)
            plt.imshow(rightforeground.astype('uint8'),alpha=0.3)
        if save:
            plt.savefig('./image/' + str(idx)+'_groud')
        else:
            plt.show()

    def ImageWithMask(self, idx, mask, left=False, right=False, save=False):
        background = np.asarray(self.sample[idx]['image'])
        plt.imshow(background[0].astype('float'),cmap='gray')
        filter = np.asarray(np.argmax(mask,axis=0))
        if left:
            leftfilter = filter == 1
            leftfilter.astype('uint8')
            leftfilter = leftfilter*255
            zerolayer = np.zeros((leftfilter.shape[0],leftfilter.shape[1]))
            leftforeground = np.stack((zerolayer,zerolayer,leftfilter),axis=-1)
            plt.imshow(leftforeground.astype('uint8'),alpha=0.3)
        if right:
            rightfilter = filter == 2
            rightfilter.astype('uint8')
            rightfilter = rightfilter*255
            zerolayer = np.zeros((rightfilter.shape[0],rightfilter.shape[1]))
            rightforeground = np.stack((rightfilter,zerolayer,zerolayer),axis=-1)
            plt.imshow(rightforeground.astype('uint8'),alpha=0.3)
        if save:
            plt.savefig('./image/' +  str(idx)+'_mask')
        else:
            plt.show()


class lungSegmentDataset(Dataset):
    def __init__(self,
                 image_path,
                 leftmask_path,
                 rightmask_path,
                 imagetransform=None,
                 labeltransform=None,
                 convert_to='L'):
        self.image_path = image_path
        self.leftmask_path = leftmask_path
        self.rightmask_path = rightmask_path
        self.imgtransform = imagetransform
        self.labtransform = labeltransform
        assert convert_to in ['RGB', 'L']
        self.convert_to = convert_to
        self.list = []

        for root, dirs, files in os.walk(image_path):
            for filename in files:
                self.list.append(filename)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_path,self.list[idx])
        left_name = os.path.join(self.leftmask_path,self.list[idx])
        right_name = os.path.join(self.rightmask_path,self.list[idx])

        img = Image.open(img_name).convert(self.convert_to)
        left = Image.open(left_name)
        right = Image.open(right_name)

        if self.imgtransform:
            img = self.imgtransform(img)
        if self.labtransform:
            left = self.labtransform(left)
            right = self.labtransform(right)

        right = right * 2
        right = right[0].type(torch.uint8)
        left = left[0].type(torch.uint8)
        label=left+right

        sample = {'image':img,'label':label}

        return sample


class LungTest(Dataset):
    def __init__(self, image_path, imgtransform, convert_to):
        self.image_path = image_path
        self.imgtransform = imgtransform
        assert convert_to in ['RGB', 'L']
        self.convert_to = convert_to
        self.list = []
        for root, dirs, files in os.walk(image_path):
            for filename in files:
                self.list.append(filename)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_path,self.list[idx])
        img = Image.open(img_name).convert(self.convert_to)
        if self.imgtransform:
            img = self.imgtransform(img)
        return {'image': img}


class JSRTBSE(Dataset):
    # The JSRT Database but with bone shadows eliminated. Gordienko used
    # this to effect in his research, so it could make sense to utilize in
    # further research
    def __init__(self, original_imgs_path, bse_imgs_path, imgtransform, convert_to, test=False):
        self.original_imgs_path = original_imgs_path
        self.bse_imgs_path = bse_imgs_path
        self.imgtransform = imgtransform
        self.convert_to = convert_to
        self.list = []
        self.test = test
        for root, dirs, files in os.walk(original_imgs_path):
            for filename in files:
                self.list.append(filename)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        orig_img_path = os.path.join(self.original_imgs_path, self.list[idx])
        arr = np.fromfile(orig_img_path, dtype='>i2').reshape((2048, 2048))
        orig_img = Image.fromarray((arr / arr.max()) * 255).convert(self.convert_to)
        if not self.test:
            bse_img_path = os.path.join(self.bse_imgs_path, self.list[idx].replace('.IMG', '.png'))
            bse = np.array(Image.open(bse_img_path))
            bse_img = Image.fromarray(((bse/bse.max()) * 255).astype('uint8')).convert(self.convert_to)
            return self.imgtransform(orig_img), self.imgtransform(bse_img)
        else:
            return self.imgtransform(orig_img)
