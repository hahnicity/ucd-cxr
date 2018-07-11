"""
loss
~~~~

Loss related functions
"""
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _assert_no_grad


def simple_undersample(loss, freq):
    """
    Downsample the loss signal by sampling it at every nth time we desire.

    :param loss: The array of loss values
    :param freq: The rate to sample loss. Example 10 would take every 10th value
    """
    new = []
    for i in range(0, len(loss), freq):
        new.append(loss[i])
    # just append the last val for posterity
    if i != len(loss) - 1:
        new.append(loss[i])
    return new


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=0.5, alpha=.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """
        :param input: Should be logit normalized predictions
        :param target: Ground truth
        """
        # XXX this method doesn't seem to be working for some reason
        # my suspicion is that it has to do with the weight.
        _assert_no_grad(target)

        t = target
        p = input
        x = input
        alpha = self.alpha
        gamma = self.gamma

        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        #w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        #w = w * (1-pt).pow(gamma)
        w = (1-pt).pow(gamma)
        w = w.detach()
        #self.register_buffer('weight', w)
        return F.binary_cross_entropy(x, t, w, size_average=False)

        #pt = input * target + (1-input)*(1-target)
        #w = self.alpha * target + (1-self.alpha)*(1-target)

        #w = w * (1-pt).pow(self.gamma)
        #w = w.detach()
        #loss = F.binary_cross_entropy(input, target, w, size_average=False)
        #return loss


class FocalLossWithLogits(torch.nn.Module):
    def __init__(self, gamma=2, alpha=.25):
        super(FocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """
        :param input: Should be logit normalized predictions
        :param target: Ground truth
        """
        p = input.sigmoid()
        # pt = p if target > 0 else 1-p
        pt = p*target + (1-p)*(1-target)
        # w = self.alpha if target > 0 else 1-self.alpha
        w = self.alpha*target + (1-self.alpha)*(1-target)
        w = w * (1-pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(input, target, w, size_average=False)
