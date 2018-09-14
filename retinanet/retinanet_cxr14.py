import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from retinanet.fpn import FPN50


class RetinaNetClassifier(nn.Module):
    def __init__(self, fpn, num_classes=14):
        super(RetinaNetClassifier, self).__init__()
        self.fpn = fpn
        self.fc = nn.Linear(256 * 5, num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        preds = []
        for idx, fm in enumerate(fms):
            fm = F.adaptive_avg_pool2d(fm, 1)
            preds.append(fm.view(fm.size(0), -1))
        return self.fc(torch.cat(preds, 1))


def retinanet_cls50(pretrained=True):
    fpn = FPN50(pretrained=pretrained)
    fpn_dict = fpn.state_dict()
    net = RetinaNetClassifier(fpn)
    # This is the method used for initializing the weights in retinanet in
    # scripts/get_state_dict.py
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.normal(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    net.fpn.load_state_dict(fpn_dict)
    return net


def test():
    net = RetinaNetClassifier(FPN50())
    preds = net(torch.autograd.Variable(torch.randn(2,3,224,224)))
    print(preds.size())
    preds_grads = torch.autograd.Variable(torch.randn(preds.size()))
    preds.backward(preds_grads)


#test()
