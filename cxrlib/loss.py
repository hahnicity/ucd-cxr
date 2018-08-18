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


class FocalLossWithAlpha(torch.nn.Module):
    def __init__(self, gamma=0.5, alpha=.75):
        super(FocalLossWithLogits, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        """
        Impl from original authors but doesn't seem to work properly. Everything ends
        up at .5 on test auc. It's probably the alpha scaling which doesn't work.

        :param input: Should be logit normalized predictions
        :param target: Ground truth
        """
        bce_loss = F.binary_cross_entropy_with_logits(input, target, size_average=False, reduce=False)
        p = input.sigmoid()
        # pt = p if target > 0 else 1-p
        pt = p*target + (1-p)*(1-target)
        # w = self.alpha if target > 0 else 1-self.alpha
        w = self.alpha*target + (1-self.alpha)*(1-target)
        w = w * (1-pt).pow(self.gamma)
        return (bce_loss * w).mean()


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        """
        Based off impl from original authors and medium post. Doesn't utilize alpha
        """
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        # this is basically just an flipped relu
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        p = F.sigmoid(input)
        # if y=0 then (1-p)^gamma, if y=1 then p^gamma
        weights = (target * p + (1-target) * (1-p)).pow(self.gamma)
        loss = weights * loss

        # XXX consider using sum
        return loss.mean()


# XXX inverse is probably not the best name for this
#
# This is really just the same thing as the module above
class InverseFocalLoss(torch.nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, input, target):
        """
        Based off impl from medium post, just with the probability coefficients flipped
        """
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        # this is basically just an flipped relu
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        p = F.sigmoid(input)
        # if y=0 then p^beta, if y=1 then (1-p)^beta
        weights = (target * (1-p) + (1-target) * p).pow(self.beta)
        loss = weights * loss

        return loss.mean()


class TailAndHeadFocalLoss(torch.nn.Module):
    def __init__(self, beta, gamma):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, input, target):
        """
        Add beta and gamma coefficients into focal loss term
        """
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        # this is basically just an flipped relu
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        p = F.sigmoid(input)
        # if y=0 then p^beta, if y=1 then (1-p)^beta
        beta_weights = (target * (1-p) + (1-target) * p).pow(self.beta)
        gamma_weights = (target * p + (1-target) * (1-p)).pow(self.gamma)
        loss = beta_weights * gamma_weights * loss

        return loss.mean()

if __name__ == "__main__":
    import numpy as np
    with torch.enable_grad():
        criterion = FocalLossWithLogits()
        input = np.random.normal(loc=0, size=(10, 10))
        input = torch.autograd.Variable(torch.FloatTensor(input), requires_grad=True)
        target = np.random.randint(0, 2, size=(10, 10))
        target = torch.autograd.Variable(torch.FloatTensor(target))
        loss = criterion(input, target)
        print('first loss: {}'.format(loss))
        loss.backward()

        input = np.random.normal(loc=0, size=(10, 10))
        input = torch.autograd.Variable(torch.FloatTensor(input), requires_grad=True)
        target = np.random.randint(0, 2, size=(10, 10))
        target = torch.autograd.Variable(torch.FloatTensor(target))
        loss = criterion(input, target)
        print('second loss: {}'.format(loss))
        loss.backward()
