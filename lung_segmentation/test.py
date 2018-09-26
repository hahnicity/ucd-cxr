import argparse
import os

import importAndProcess as iap
import model
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from cxrlib.models.unet_models import unet11, unet16

parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('-m', '--model', choices=['unet11', 'unet16', 'resnet'])
parser.add_argument('-o', '--out-dir', default='images/')
parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint', required=True)
parser.add_argument('--non-montgomery', action='store_true', help='toggle this flag if you are working on a non-montgomery dataset')
args = parser.parse_args()

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

if args.model == 'resnet':
    segNet = model.segmentNetwork().cuda()
    resize_dim = (400, 400)
    convert_to = 'L'
elif args.model == 'unet11':
    segNet = unet11(out_filters=3).cuda()
    resize_dim = (224, 224)
    convert_to = 'RGB'
elif args.model == 'unet16':
    segNet = unet16(out_filters=3).cuda()
    resize_dim = (224, 224)
    convert_to = 'RGB'

dataset = iap.LungSegmentTest(
    args.img_path,
    Compose([Resize(resize_dim),ToTensor(),normalize]),
    convert_to=convert_to
)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

segNet = torch.nn.DataParallel(segNet)
segNet.load_state_dict(torch.load(args.resume_from))
show = iap.visualize(dataset)

for i,sample in enumerate(dataloader):
    img = torch.autograd.Variable(sample['image']).cuda()
    mask = segNet(img)
    if not args.non_montgomery:
        show.ImageWithGround(i,True,True,save=True)
    show.ImageWithMask(i,mask.cpu().detach().numpy()[0],True,True,save=True)
