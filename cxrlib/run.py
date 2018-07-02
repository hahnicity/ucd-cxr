"""
run
~~~

Contains tools that we can use across the project to run our dataset.
"""
from time import time

import numpy as np
import torch

from cxrlib.results import compute_AUCs


class RunModel(object):
    def __init__(self, args, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, use_cuda, reporting):
        """
        :param args: arguments from command line. Can be None if we're not using any
        :param model: pytorch model to train
        :param ...: TODO
        :param ...: TODO
        :param ...: TODO
        :param ...: TODO
        :param ...: TODO
        :param use_cuda: True if we are using a GPU, False otherwise
        :param reporting: an instance of cxrlib.results.Reporting
        """
        self.args = args
        try:
            self.print_progress = self.args.print_progress
        except:
            self.print_progress = False
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.reporting = reporting
        self.cuda_wrapper = lambda x: x.cuda() if use_cuda else x
        self.cuda_async_wrapper = lambda x: x.cuda(non_blocking=True) if use_cuda else x

    def train_multi_epoch(self, epochs):
        """
        Train for multiple epochs

        :param epochs: number of epochs to train for
        """
        for ep in range(epochs):
            self.generic_train_epoch(ep+1)

    def generic_train_epoch(self, epoch_num):
        self.model.train()
        with torch.enable_grad():
            for i, (inp, target) in enumerate(self.train_loader):
                self.pre_batch_actions()

                batch_start = time()

                target = self.cuda_async_wrapper(target)
                inp = self.cuda_async_wrapper(inp)
                bs, c, h, w = inp.size()
                target = torch.autograd.Variable(target)
                input_var = torch.autograd.Variable(inp.view(-1, c, h, w))
                output = self.model(input_var)

                self.optimizer.zero_grad()
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                batch_time = time() - batch_start
                self.reporting.update('train_loss', loss)
                self.reporting.update('batch_time', batch_time)

                if self.print_progress:
                    print('Epoch: {}, Batch: {}/{}, Batch time: {}, Loss: {}\r'.format(
                        epoch_num, i, len(self.train_loader), batch_time, str(self.reporting.get_meter('train_loss'))
                    ), end="")

                self.post_batch_actions()

            self.post_epoch_actions()

    def generic_test_epoch(self):
        """
        Run a test epoch. At the end, returns all batch predictions and ground truth
        labels.
        """
        self.model.eval()
        gt = None
        pred = None
        # does not work on torch < .4
        with torch.no_grad():
            for i, (inp, target) in enumerate(self.test_loader):
                bs, c, h, w = inp.size()
                target = self.cuda_async_wrapper(target)
                # This does not support data augmentation like kfold. In the future
                # we need to support this
                input_var = self.cuda_async_wrapper(torch.autograd.Variable(inp.view(-1, c, h, w)))
                output = self.model(input_var)
                if pred is None:
                    pred = output.data
                    gt = target
                else:
                    pred = torch.cat((pred, output.data), 0)
                    gt = torch.cat((gt, target), 0)

                if self.print_progress:
                    print("")
                    print('Eval Test Batch: {}/{}\r'.format(i, len(self.test_loader)), end="")

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        self.reporting.update('test_auc', AUROC_avg)

    def pre_batch_actions(self):
        """
        Stub methods that can be utilized for customizable actions
        """
        pass

    def post_batch_actions(self):
        """
        Stub methods that can be utilized for customizable actions
        """
        pass

    def post_epoch_actions(self):
        """
        Stub methods that can be utilized for customizable actions
        """
        pass
