"""
Try to analyze specific things that might be contributing to loss.

One thing that might be interesting is analyzing a specific class to see how
the loss distribution of positive and negative samples changes over epochs
This in particular would measure direct class loss, but you could also measure indirect
loss where you are looking for classes that get confused with other classes
"""
import glob
import os

import numpy as np
import torch

from cxrlib.constants import CLASS_NAMES


RESULTS_DIR = 'results/loss_analysis'

def main():
    results_dir = os.path.join(os.path.dirname(__file__), RESULTS_DIR)
    loss_files = glob.glob(os.path.join(results_dir, '*loss.pt'))
    gt_files = glob.glob(os.path.join(results_dir, '*gt.pt'))

    sorted_loss_files = sorted(loss_files, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[3])))
    sorted_gt_files = sorted(gt_files, key=lambda x: (int(x.split('-')[1]), int(x.split('-')[3])))
    loss_by_epoch = []
    gt_by_epoch = []
    last_epoch = -1

    for i, f in enumerate(sorted_loss_files):
        loss = torch.load(f).cpu()
        gt = torch.load(sorted_gt_files[i]).cpu()
        epoch = int(f.split('-')[1])
        if epoch != last_epoch:
            loss_by_epoch.append(loss)
            gt_by_epoch.append(gt)
        else:
            loss_by_epoch[-1] = torch.cat((loss_by_epoch[-1], loss), 0)
            gt_by_epoch[-1] = torch.cat((gt_by_epoch[-1], gt), 0)
        last_epoch = epoch

    loss_by_epoch_by_class = []
    for epoch in range(len(loss_by_epoch)):
        loss_by_epoch_by_class.append({})
        for class_idx in range(len(CLASS_NAMES)):
            epoch_loss = loss_by_epoch[epoch].numpy()
            epoch_gt = gt_by_epoch[epoch].numpy()
            # The mask will give us the row idx, afterwards we need
            # to extract the exact cols
            class_mask = epoch_gt[:, class_idx] == 1
            class_loss = epoch_loss[class_mask, class_idx]
            non_class_loss = epoch_loss[np.logical_not(class_mask), class_idx]

            class_name = CLASS_NAMES[class_idx]
            loss_by_epoch_by_class[-1][class_name] = {
                'class_loss': class_loss,
                'non_class_loss': non_class_loss,
            }
    import IPython; IPython.embed()


if __name__ == "__main__":
    main()
