from copy import deepcopy

import torch


class FastGeometricEnsemble(object):
    def __init__(self, clr, model, epoch_start):
        """
        Class for creating a fast geometric ensemble utilizing cyclic learning rates

        :param clr: an instance of CyclicLR. Garipov uses the triangular mode for his experiments
        :param model: NN we are using
        :param epoch_start: Approximate location to start the FGE, although it will
                            not be exact because FGE only ensembles at cycle minimums
        """
        self.ensemble = torch.nn.Module()
        self.epoch_start = epoch_start
        self.clr = clr
        self.min_lr = clr.base_lrs[0]
        self.epoch_num = 1
        self.n_models = 0
        self.model = model

    def batch_step(self):
        """
        Check where we are on the cyclical loss curve. If we are at the minimum
        cyclic loss then create a new ensemble model.
        """
        if self.epoch_num >= self.epoch_start and self.clr.get_lr()[0] == self.min_lr:
            self.n_models += 1
            self.ensemble.add_module("fge{}".format(self.n_models), deepcopy(self.model))

    def epoch_step(self):
        self.epoch_num += 1

    def get_ensemble(self):
        return self.ensemble
