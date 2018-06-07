"""
Implementation of Stochastic Weight Averaging. Code Shamelessly taken from Tim Garipov
at https://github.com/timgaripov/swa
"""
import torch


class SWA(object):
    def __init__(self, non_swa_model, swa_model, epoch_start, loader, total_epochs, device='cpu'):
        """
        Implement Stochastic Weight Averaging for non-cyclic learning rates
        """
        self.non_swa_model = non_swa_model
        self.swa_model = swa_model
        self.epoch_start = epoch_start
        self.loader = loader
        self.total_epochs = total_epochs
        self.device = device
        self.epoch_num = 1
        self.n_models = 0

    def is_swa_training(self):
        if self.epoch_num >= self.epoch_start:
            return True
        else:
            return False

    def step(self):
        if self.epoch_num >= self.epoch_start:
            self.update_weights()
            self.n_models += 1
        if self.epoch_num == self.total_epochs:
            self.bn_update()
        self.epoch_num += 1

    def update_weights(self):
        for non_swa_param, swa_param in zip(self.non_swa_model.parameters(), self.swa_model.parameters()):
            swa_param.data = ((swa_param.data * self.n_models) + non_swa_param.data) / (self.n_models + 1)

    def _check_bn(self, module, flag):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            flag[0] = True

    def check_bn(self):
        flag = [False]
        self.swa_model.apply(lambda module: self._check_bn(module, flag))
        return flag[0]

    def reset_bn(self, module):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)

    def _get_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum

    def _set_momenta(self, module, momenta):
        if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
            module.momentum = momenta[module]

    def bn_update(self):
        """
        BatchNorm buffers update (if any).
        Performs 1 epochs to estimate buffers average using train dataset.
        """
        if not self.check_bn():
            return
        self.swa_model.train()
        momenta = {}
        self.swa_model.apply(self.reset_bn)
        self.swa_model.apply(lambda module: self._get_momenta(module, momenta))
        n = 0
        for i, (input, _) in enumerate(self.loader):
            if self.device == 'cuda':
                input = input.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            b = input_var.data.size(0)

            momentum = b / (n + b)
            for module in momenta.keys():
                module.momentum = momentum

            self.swa_model(input_var)
            n += b

        self.swa_model.apply(lambda module: self._set_momenta(module, momenta))
