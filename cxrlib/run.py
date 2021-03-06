"""
run
~~~

Contains tools that we can use across the project to run our dataset.
"""
from time import time

import numpy as np
import torch
import torch.nn.functional as F

from cxrlib.results import compute_AUCs


class RunModel(object):
    def __init__(self, args, model, train_loader, test_loader, optimizer, lr_scheduler, criterion, use_cuda, reporting, validation_loader=None, tmp_save_path='/tmp'):
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
        :param validation_loader: Loader for validation set. Doesn't have to be set
        :param tmp_save_path: Path to save your model after an epoch ends
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
        self.validation_loader = validation_loader
        self.epoch = 0
        # XXX Need to cleanup everything in this path if model executes ok
        self.tmp_save_path = tmp_save_path

    def train_multi_epoch(self, epochs):
        """
        Train for multiple epochs

        :param epochs: number of epochs to train for
        """
        self.pre_train_actions()
        for ep in range(epochs):
            self.generic_train_epoch(ep+1)
            if self.validation_loader:
                self.generic_validation_epoch()

    def generic_train_epoch(self, epoch_num):
        self.epoch = epoch_num
        self.model.train()
        with torch.enable_grad():
            for i, (inp, target) in enumerate(self.train_loader):
                self.pre_batch_actions()

                batch_start = time()

                target = self.cuda_async_wrapper(target)
                inp = self.cuda_async_wrapper(inp)
                target = torch.autograd.Variable(target)
                # If we are performing multi-cropping then we will preserve batch
                # size by iterating on each crop individually. This does have the
                # possible effect of overtraining tho. Dunno how to get around this
                # and still not run out of memory.
                if inp.ndimension() == 5:
                    bs, crops, c, h, w = inp.size()
                    for k in range(crops):
                        crop_inp = torch.autograd.Variable(inp[:,k,:,:,:])
                        output = self.model(crop_inp)
                        crop_target = target[:,k,:]

                        self.optimizer.zero_grad()
                        loss = self.criterion(output, crop_target)
                        loss.backward()
                        self.optimizer.step()
                        self.reporting.update('train_loss', loss)
                else:
                    inp = torch.autograd.Variable(inp)
                    output = self.model(inp)

                    self.optimizer.zero_grad()
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    self.reporting.update('train_loss', loss)

                batch_time = round(time() - batch_start, 4)
                self.reporting.update('batch_time', batch_time)
                del loss

                if self.print_progress:
                    print('Epoch: {}, Batch: {}/{}, Batch time: {}, Loss: {}\r'.format(
                        epoch_num, i+1, len(self.train_loader), batch_time, str(self.reporting.get_meter('train_loss'))
                    ), end="")

                self.post_batch_actions()

            self.reporting.update('loss_rate', self.optimizer.state_dict()['param_groups'][0]['lr'])
            self.reporting.save('model', dir_override=self.tmp_save_path)
            self.post_epoch_actions()

    def generic_validation_epoch(self):
        """
        Run an epoch on the validation set.
        """
        self.model.eval()
        self.reporting.new_unsaved_meter('validation_epoch_loss')
        gt = None
        pred = None
        with torch.no_grad():
            for i, (inp, target) in enumerate(self.validation_loader):
                target = self.cuda_async_wrapper(target)
                inp = self.cuda_async_wrapper(inp)
                target = torch.autograd.Variable(target)

                if inp.ndimension() == 5:
                    bs, crops, c, h, w = inp.size()
                    for k in range(crops):
                        crop_inp = torch.autograd.Variable(inp[:,k,:,:,:])
                        output = self.model(crop_inp)
                        crop_target = target[:,k,:]

                        loss = self.criterion(output, crop_target)
                        self.reporting.update('validation_loss', loss)
                        self.reporting.update('validation_epoch_loss', loss)

                        if pred is None:
                            pred = output.data
                            gt = crop_target
                        else:
                            pred = torch.cat((pred, output.data), 0)
                            gt = torch.cat((gt, crop_target), 0)
                else:
                    inp = torch.autograd.Variable(inp)
                    output = self.model(inp)
                    # XXX should I evaluate based on averaging AUC over all crops
                    # like we do in the testing phase??
                    loss = self.criterion(output, target)
                    self.reporting.update('validation_loss', loss)
                    self.reporting.update('validation_epoch_loss', loss)

                    if pred is None:
                        pred = output.data
                        gt = target
                    else:
                        pred = torch.cat((pred, output.data), 0)
                        gt = torch.cat((gt, target), 0)

        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        if self.print_progress:
            print("")
            print("Validation AUC: {}".format(AUROC_avg))
        self.reporting.update('validation_auc', AUROC_avg)
        self.post_validation_actions()

    def _test_epoch(self):
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
                inp = self.cuda_async_wrapper(torch.autograd.Variable(inp.view(-1, c, h, w)))
                # XXX eventually output will need to handle multi-crop
                #
                # XXX actually we may have been doing testing incorrectly because
                # we weren't applying sigmoid... sigh
                output = self.model(inp)
                if pred is None:
                    pred = output.data
                    gt = target
                else:
                    pred = torch.cat((pred, output.data), 0)
                    gt = torch.cat((gt, target), 0)
        return gt, pred

    def generic_test_epoch(self):
        """
        Run a test epoch. At the end, returns all batch predictions and ground truth
        labels.
        """
        gt, pred = self._test_epoch()
        AUROCs = compute_AUCs(gt, pred)
        AUROC_avg = np.array(AUROCs).mean()
        self.reporting.update('test_auc', AUROC_avg)
        if self.print_progress:
            print("\nTest AUC: {}\n".format(AUROC_avg))

    def pre_train_actions(self):
        """
        Stub methods that can be utilized for customizable actions
        """
        pass

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

    def post_validation_actions(self):
        """
        Stub methods that can be utilized for customizable actions after all validation data is
        evaluated
        """
        pass


class RunModelWithValidationLRScheduler(RunModel):
    """
    This assumes you have a validation set and you are using the
    ReduceLROnPlateau loss scheduler.
    """
    def post_validation_actions(self):
        epoch_loss = self.reporting.get_meter('validation_epoch_loss').values
        mean_loss = epoch_loss.mean()
        self.lr_scheduler.step(mean_loss)


class RunModelWithTestAUCReporting(RunModel):
    """
    Run your model and your testing set at the end of each epoch
    """
    def post_epoch_actions(self):
        try:
            run_after_ep = self.args.run_test_after_epoch
        except:
            run_after_ep = 0
        if self.epoch > run_after_ep:
            self.generic_test_epoch()


class RunModelWithAUCAndValLR(RunModelWithValidationLRScheduler, RunModelWithTestAUCReporting):
    pass
