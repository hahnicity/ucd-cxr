"""
init
~~~~

An extension of torch.nn.init and just provides convenience. We use normal
distribution for initialization for xavier and kaiming because ImageNet
weights are much closer to a normal distribution than a uniform

Usage:

    from cxrlib.init import kaiming_init

    model = MyModel()
    model.apply(kaiming_init)
"""
import numpy as np
from scipy.stats import t as student_t

import torch
from torch.nn.init import (
    _calculate_correct_fan,
    _calculate_fan_in_and_fan_out,
    kaiming_normal_,
    xavier_normal_
)


def kaiming_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        kaiming_normal_(m.weight.data)
        if m.bias is not None:
            try:
                _calculate_fan_in_and_fan_out(m.bias.data)
            except ValueError:
                pass
            else:
                kaiming_normal_(m.bias.data)


def xavier_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        xavier_normal_(m.weight.data)
        if m.bias is not None:
            try:
                _calculate_fan_in_and_fan_out(m.bias.data)
            except ValueError:
                pass
            else:
                xavier_normal_(m.bias.data)


def student_t_init(m, v=10):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        rand = student_t.rvs(v, size=m.weight.numel())
        # normalize between -1 and 1
        rand /= np.max(np.abs(rand), axis=0)
        m.weight.data = torch.FloatTensor(rand).view(m.weight.size())
