from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import encoder
from utils import box_iou, kaggle_iou, one_hot_embedding
from torch.autograd import Variable


class FocalLoss(nn.Module):
    def __init__(self, num_classes=20):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        # alpha is used as a weighting factor by the group. They say it improves
        # results compared to if alpha was just set to 1
        alpha = 0.25
        gamma = 2.0

        # setup a 1 hot embedding of the ground truth so that it fits nicely
        # into BCE
        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)  # [N,21]
        t = t[:,1:]  # exclude background
        t = Variable(t).cuda()  # [N,20]

        # Here is where the actual FC is computed
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = alpha*t + (1-alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1-pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        # I think gamma is implicitly set to 2 here

        t = one_hot_embedding(y.data.cpu(), 1+self.num_classes)
        t = t[:,1:]
        t = Variable(t).cuda()

        xt = x*(2*t-1)  # xt = x if t > 0 else -x
        pt = (2*xt+1).sigmoid()

        w = alpha*t + (1-alpha)*(1-t)
        loss = -w*pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        masked_cls_preds = cls_preds[mask].view(-1,self.num_classes)
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])
        #cls_loss = self.focal_loss(masked_cls_preds, cls_targets[pos_neg])

        num_pos = pos.sum().float()

        print('loc_loss: {} | cls_loss: {}\r'.format(np.round(loc_loss.detach()/num_pos, 4), np.round(cls_loss.detach()/num_pos, 4)), end='')

        if num_pos != 0:
            loss = (loc_loss+cls_loss)/num_pos
        else:
            loss = loc_loss+cls_loss
        return loss


class BasicLoss(nn.Module):
    def __init__(self):
        super(BasicLoss, self).__init__()
        self.num_classes = 1

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        masked_loc_preds = loc_preds[pos].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[pos].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        pos_neg = cls_targets > -1  # exclude ignored anchors

        # cls_loss ends up skewing way too much toward the background class.
        # It would probably take very many epochs until the loss was very close
        # to 0 to get any solid predictions if we just left this alone
        cls_loss = F.binary_cross_entropy_with_logits(cls_preds[pos_neg].squeeze(), cls_targets[pos_neg].float())
        return (cls_loss + loc_loss)


class StatLoss(nn.Module):
    # Basic loss function with Precision + False Pos loss function to speed up
    # prediction of positive bounding boxes
    def __init__(self):
        super(StatLoss, self).__init__()
        self.num_classes = 1

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        masked_loc_preds = loc_preds[pos].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[pos].view(-1,4)  # [#pos,4]
        loc_loss = 2*F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        pos_neg = cls_targets > -1  # exclude ignored anchors

        # cls_loss ends up skewing way too much toward the background class.
        # It would probably take very many epochs until the loss was very close
        # to 0 to get any solid predictions if we just left this alone
        cls_loss = F.binary_cross_entropy_with_logits(cls_preds[pos_neg].squeeze(), cls_targets[pos_neg].float())

        # XXX this isn't working because the easiest way to maximize tps is to not
        # predict any positives. So then, how do you actually predict a positive, while still
        # balancing for not predicting false positives?
        #
        # The thing is, a loss function discusses what you do poorly. so a tp loss literally
        # does nothing, it only says good job if you get it right, and doesn't say anything
        # if you get it wrong. In fact by mixing tps with fps, you actually dilute the effect
        # you would get from a fn predictor
        pred_pos = cls_preds[pos]
        if pred_pos.size() == (1, 1):
            pred_pos = pred_pos[0]
        else:
            pred_pos = pred_pos.squeeze()

        # precision loss
        if len(pred_pos) != 0:
            precision_loss = 1.1*F.binary_cross_entropy_with_logits(pred_pos, cls_targets[pos].float())
        else:
            precision_loss = 0

        # this will pull only fps
        fp = ((cls_preds.sigmoid() > .5).squeeze() & (cls_targets == 0)) > 0
        false_pos = cls_preds[fp]
        if false_pos.size() == (1, 1):
            false_pos = false_pos[0]
        else:
            false_pos = false_pos.squeeze()

        # fp loss
        if len(false_pos) == 0:
            fp_cls_loss = 0
        else:
            fp_cls_loss = F.binary_cross_entropy_with_logits(false_pos, cls_targets[fp].float())

        return (cls_loss + loc_loss + precision_loss + fp_cls_loss)

class StatLossV2(nn.Module):
    # Basic loss function with Precision + False Pos loss function to speed up
    # prediction of positive bounding boxes
    def __init__(self):
        super(StatLossV2, self).__init__()
        self.num_classes = 1

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        masked_loc_preds = loc_preds[pos].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[pos].view(-1,4)  # [#pos,4]
        loc_loss = 2*F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)

        pos_neg = cls_targets > -1  # exclude ignored anchors

        # cls_loss ends up skewing way too much toward the background class.
        # It would probably take very many epochs until the loss was very close
        # to 0 to get any solid predictions if we just left this alone
        cls_loss = F.binary_cross_entropy_with_logits(cls_preds[pos_neg].squeeze(), cls_targets[pos_neg].float())

        # XXX this isn't working because the easiest way to maximize tps is to not
        # predict any positives. So then, how do you actually predict a positive, while still
        # balancing for not predicting false positives?
        #
        # The thing is, a loss function discusses what you do poorly. so a tp loss literally
        # does nothing, it only says good job if you get it right, and doesn't say anything
        # if you get it wrong. In fact by mixing tps with fps, you actually dilute the effect
        # you would get from a fn predictor
        pred_pos = cls_preds[pos]
        if pred_pos.size() == (1, 1):
            pred_pos = pred_pos[0]
        else:
            pred_pos = pred_pos.squeeze()

        # precision loss
        if len(pred_pos) != 0:
            precision_loss = F.binary_cross_entropy_with_logits(pred_pos, cls_targets[pos].float())
        else:
            precision_loss = 0

        # this will pull only fps
        fp = ((cls_preds.sigmoid() > .5).squeeze() & (cls_targets == 0)) > 0
        false_pos = cls_preds[fp]
        if false_pos.size() == (1, 1):
            false_pos = false_pos[0]
        else:
            false_pos = false_pos.squeeze()

        # fp loss
        if len(false_pos) == 0:
            fp_cls_loss = 0
        else:
            fp_cls_loss = F.binary_cross_entropy_with_logits(false_pos, cls_targets[fp].float())

        return (cls_loss + loc_loss + precision_loss + fp_cls_loss)

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
        self.num_classes = 1
        self.encoder = encoder.DataEncoder()
        self.input_size = 224

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]

        mask = pos.unsqueeze(2).expand_as(loc_preds)       # [N,#anchors,4]
        masked_loc_preds = loc_preds[mask].view(-1,4)      # [#pos,4]
        masked_loc_targets = loc_targets[mask].view(-1,4)  # [#pos,4]
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
