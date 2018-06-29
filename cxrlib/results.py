import os

from sklearn.metrics import roc_auc_score
import torch

from cxrlib import constants


class SavedObjects(object):
    def __init__(self, file_dir):
        """
        Because saving objects after a network finishes training is tricky,
        we can just use this helper class to keep track of the objects we
        want to save. Afterwards save everything to file. Example:

            saved_objs = SavedObjects('/path/to/results')
            model = ResNet50()
            training_loss = []
            saved_objs.register(model, 'resnet50_weights', True)
            saved_objs.register(training_loss, 'train_loss', False)

            ... Do training stuff
            ... Do testing stuff

            saved_objs.save_all('date_finished')
        """
        self.saved_objects = []
        self.file_dir = file_dir

    def register(self, obj, file_prefix, save_weights):
        """
        :param obj: object you want to save later
        :param file_prefix: prefix of file to save eg. "model_weights"
        :param save_weights: True if its a nn model and we only want to save weights.
                             False otherwise. We do this so we only save model weights
                             and not the entire model
        """
        self.saved_objects.append((obj, file_prefix, save_weights))

    def save_all(self, file_suffix):
        for obj, prefix, save_weights in self.saved_objects:
            filename = "_".join([prefix, file_suffix]) + ".pt"
            filepath = os.path.join(self.file_dir, filename)
            if save_weights:
                torch.save(obj.state_dict(), filepath)
            else:
                torch.save(obj, filepath)

# XXX A good idea would be to make an overarching results tabulator that can handle
# multiple streams of results. The deal here is that it would need to keep track of
# epoch and batch steps and then it could fill in results for both of these
# as new results are updated.
#
# So doing the batch level and epoch level work is a bit more difficult, so maybe
# think of a way to come back to that. What is reasonable currently is a series
# of pre-made meters for loss, timing, etc

class Reporting(SavedObjects):
    def __init__(self, file_dir):
        super(Reporting, self).__init__(file_dir)
        self.meters = {
            "train_loss": Meter('train_loss'),
            "batch_time": Meter('batch_time'),
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
        :param name: meter name
        """
        self.meters[name] = Meter(name)
        self.register(self.meters[name])

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
