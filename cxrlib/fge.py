from copy import deepcopy

import torch

# So I have a few ideas on how I can do this.
#
# 1. Just do majority rules
# 2. Create a mean of the predictions and throw it into sigmoid again
# 3. Add in an additional FC layer that learns to weight different predictions
#    from different models. This would require some re-training on some epochs
# 4. Do like in Siamese Networks where strip the FC and sigmoid layers from
#    constituent models, throw in a more comprehensive FC layer for all 3 models
#    then retrain
class Ensemble(torch.nn.Module):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x):
        # For now just do mean of predictions. (and sigmoid??)
        #
        # They've already gone through sigmoids, so why would another
        # sigmoid make sense?
        result = None
        n_models = len(self._modules.keys())
        for module in self.children():
            if result is None:
                result = module(x)
            else:
                result = torch.cat([result, module(x)])
        batch_size = x.size(0)
        return result.view(batch_size, n_models, -1).mean(1)


class FastGeometricEnsemble(object):
    def __init__(self, clr, model, epoch_start):
        """
        Class for creating a fast geometric ensemble utilizing cyclic learning rates

        :param clr: an instance of CyclicLR. Garipov uses the triangular mode for his experiments
        :param model: NN we are using
        :param epoch_start: Approximate location to start the FGE, although it will
                            not be exact because FGE only ensembles at cycle minimums
        """
        self.ensemble = Ensemble()
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
            if isinstance(self.model, torch.nn.DataParallel):
                self.ensemble.add_module("fge{}".format(self.n_models), deepcopy(self.model.module))
            else:
                self.ensemble.add_module("fge{}".format(self.n_models), deepcopy(self.model))

    def epoch_step(self):
        self.epoch_num += 1

    def get_ensemble(self):
        return self.ensemble
