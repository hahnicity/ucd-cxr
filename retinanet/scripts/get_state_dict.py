'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''
import argparse
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet50

from retinanet.fpn import FPN50
from retinanet.retinanet import RetinaNet

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-classes', type=int, default=20)
parser.add_argument('--model', help='Pretrained weights for resnet50 saved to some file')
parser.add_argument('-o', '--output', default='net.pth')
args = parser.parse_args()

print('Loading pretrained ResNet50 model..')
if not args.model:
    d = resnet50(pretrained=True).state_dict()
else:
    try:
        model = torch.load(args.model).module.model
    except:
        model = torch.load(args.model).module.fpn
    d = model.state_dict()

print('Loading into FPN50..')
fpn = FPN50()
dd = fpn.state_dict()
for k in d.keys():
    if not k.startswith('fc'):  # skip fc layers
        dd[k] = d[k]

print('Saving RetinaNet..')
net = RetinaNet(num_classes=args.num_classes)
for m in net.modules():
    if isinstance(m, nn.Conv2d):
        init.normal(m.weight, mean=0, std=0.01)
        if m.bias is not None:
            init.constant(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

pi = 0.01
init.constant(net.cls_head[-1].bias, -math.log((1-pi)/pi))

net.fpn.load_state_dict(dd)
torch.save(net.state_dict(), args.output)
print('Done!')
