import argparse
import os

import importAndProcess as iap
import model
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

parser = argparse.ArgumentParser()
parser.add_argument('img_path')
parser.add_argument('-o', '--out-dir', default='images/')
parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint', required=True)
parser.add_argument('--non-montgomery', action='store_true', help='toggle this flag if you are working on a non-montgomery dataset')
args = parser.parse_args()

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
dataset = iap.LungSegmentTest(
    args.img_path, Compose([Resize((400,400)),ToTensor(),normalize]),
)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

segNet = model.segmentNetwork().cuda()
segNet = torch.nn.DataParallel(segNet)
segNet.load_state_dict(torch.load(args.resume_from))
show = iap.visualize(dataset)

for i,sample in enumerate(dataloader):
    img = torch.autograd.Variable(sample['image']).cuda()
    mask = segNet(img)
    if not args.non_montgomery:
        show.ImageWithGround(i,True,True,save=True)
    show.ImageWithMask(i,mask.cpu().detach().numpy()[0],True,True,save=True)
