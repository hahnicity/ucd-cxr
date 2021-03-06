import os

from sklearn.metrics import roc_auc_score
import torch

from cxrlib import constants


class SavedObjects(object):
    def __init__(self, file_dir, file_suffix):
        """
        Because saving objects after a network finishes training is tricky,
        we can just use this helper class to keep track of the objects we
        want to save. Afterwards save everything to file. Example:

            saved_objs = SavedObjects('/path/to/results', 'file_suffix')
            model = ResNet50()
            training_loss = []
            saved_objs.register(model, 'resnet50_weights', True)
            saved_objs.register(training_loss, 'train_loss', False)

            ... Do training stuff
            ... Do testing stuff

            saved_objs.save_all()
        """
        self.saved_objects = {}
        self.file_dir = file_dir
        self.file_suffix = file_suffix

    def register(self, obj, file_prefix, save_weights):
        """
        :param obj: object you want to save later
        :param file_prefix: prefix of file to save eg. "model_weights"
        :param save_weights: True if its a nn model and we only want to save weights.
                             False otherwise. We do this so we only save model weights
                             and not the entire model
        """
        self.saved_objects[file_prefix] = (obj, save_weights)

    def save(self, name, timestamp="", dir_override=None):
        obj, save_weights = self.saved_objects[name]
        joined = [name, self.file_suffix, timestamp] if timestamp else [name, self.file_suffix]
        filename = "_".join(joined) + ".pt"
        if not dir_override:
            filepath = os.path.join(self.file_dir, filename)
        else:
            filepath = os.path.join(dir_override, filename)

        if save_weights:
            torch.save(obj.state_dict(), filepath)
        else:
            torch.save(obj, filepath)

    def save_all(self, timestamp=""):
        for name in self.saved_objects:
            self.save(name, timestamp=timestamp)


class Reporting(SavedObjects):
    def __init__(self, file_dir, file_suffix):
        super(Reporting, self).__init__(file_dir, file_suffix)
        self.meters = {
            "train_loss": Meter('train_loss'),
            'validation_loss': Meter('validation_loss'),
            'loss_rate': Meter('loss_rate'),
            "batch_time": Meter('batch_time'),
            'validation_auc': Meter('validation_auc'),
            'test_auc': Meter('test_auc'),
        }
        for name, meter in self.meters.items():
            self.register(meter, name, False)

    def get_meter(self, name):
        """
        :param name: meter name
        """
        return self.meters[name]

    def new_meter(self, name):
        """
        Create a new quantitative meter that will be registered to be saved

        :param name: meter name
        """
        self.meters[name] = Meter(name)
        self.register(self.meters[name], name, False)

    def new_unsaved_meter(self, name):
        """
        Create a new meter that will not be saved

        :param name: meter name
        """
        self.meters[name] = Meter(name)

    def update(self, meter, val):
        self.meters[meter].update(val)


class Meter(object):
    """
    A little helper class which keeps track of statistics during an epoch.
    """
    def __init__(self, name, cumulative=False):
        self.cumulative = cumulative
        if type(name) == str:
            name = (name,)
        self.name = name
        self.values = torch.FloatTensor([])
        self._total = torch.zeros(len(self.name))
        self._last_value = torch.zeros(len(self.name))
        self._count = 0.0

    def update(self, data, n=1):
        self._count = self._count + n
        if isinstance(data, torch.autograd.Variable):
            self._last_value.copy_(data.data)
            self.values = torch.cat((self.values, data.data.cpu().view(1)), 0)
        elif isinstance(data, torch.Tensor):
            self._last_value.copy_(data)
            self.values = torch.cat((self.values, data.cpu().view(1)), 0)
        else:
            self._last_value.fill_(data)
            self.values = torch.cat((self.values, torch.FloatTensor([data])), 0)
        self._total.add_(self._last_value)

    def value(self):
        if self.cumulative:
            return self._total
        else:
            return self._total / self._count

    def __repr__(self):
        return '\t'.join(['%s: %.5f (%.3f)' % (n, lv, v)
            for n, lv, v in zip(self.name, self._last_value, self.value())])


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(len(constants.CLASS_NAMES)):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs
