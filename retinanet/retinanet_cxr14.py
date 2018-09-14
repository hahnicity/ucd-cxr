import torch
import torch.nn as nn
import torch.nn.functional as F

from fpn import FPN50


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


def test():
    net = RetinaNetClassifier(FPN50())
    preds = net(torch.autograd.Variable(torch.randn(2,3,224,224)))
    print(preds.size())
    preds_grads = torch.autograd.Variable(torch.randn(preds.size()))
    preds.backward(preds_grads)


test()
