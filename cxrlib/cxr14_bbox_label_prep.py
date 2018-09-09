"""
cxr14_bbox_label_prep
~~~~~~~~~~~~~~~~~~~~~~~

Process the existing bbox file into something more standard and create
new split of data. In new dataset train and test sets will be merged
together and validation set will be utilized for model validation. Some
samples from test set with bounding boxes will be moved into validation
set. New test set will largely be defined by character of existing validation
set, so ensure that the validation set is sufficiently constructed before
progressing with this.
"""
import argparse
import csv
import os

import numpy as np
import pandas as pd

from cxrlib.constants import CLASS_NAMES
from cxrlib.label_preprocessing import knapsack_splitter

TRAIN_TEST_SPLIT = .8


def add_bbox_annos_to_bbox_set(bbox_df, final_list):
    base_n_cols = len(bbox_df.columns)
    for idx, row in bbox_df.iterrows():
        for bbox_annos in final_list:
            if bbox_annos[0] == row[0]:
                for offset, val in enumerate(bbox_annos[1:]):
                    bbox_df.loc[idx, base_n_cols+offset] = val
                break
        else:
            raise Exception('you have a problem')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bbox_filepath')
    parser.add_argument('train_labels_path')
    parser.add_argument('val_labels_path')
    parser.add_argument('test_labels_path')
    args = parser.parse_args()

    final_list = []
    img_index = dict()

    # cleanup bounding box annotations and put them into a standardized state
    with open(args.bbox_filepath, "r") as f:
        reader = csv.reader(f)
        for idx, l in enumerate(reader):
            if idx == 0:
                continue
            # x, y are upper left corner
            img, cls, x, y, w, h = l
            if cls == "Infiltrate":
                cls = "Infiltration"
            cls_idx = CLASS_NAMES.index(cls)
            if img in img_index:
                bbox_line = final_list[img_index[img]]
            else:
                bbox_line = [img]

            x_min = float(x)
            y_min = float(y)
            x_max = x_min + float(w)
            y_max = y_min + float(h)
            if y_max > 1024:
                print('you got a x problem: line {}'.format(idx))

            bbox_line.extend([x_min, y_min, x_max, y_max, cls_idx])
            if img in img_index:
                final_list[img_index[img]] = bbox_line
            else:
                final_list.append(bbox_line)
                img_index[img] = idx

    # Now perform dataset splitting
    #
    # all bbox obs are located in test set, so we will need to pick patients
    # with bbox obs, move them into train set, and then in exchange move patients
    # from train to test set.
    train_labels = pd.read_csv(args.train_labels_path, header=None, sep=' ').dropna(axis=1)
    val_labels = pd.read_csv(args.val_labels_path, header=None, sep=' ').dropna(axis=1)
    test_labels = pd.read_csv(args.test_labels_path, header=None, sep=' ').dropna(axis=1)
    n_cols = len(train_labels.columns)
    train_labels.rename(columns=dict(zip(train_labels.columns, range(n_cols))), inplace=True)
    val_labels.rename(columns=dict(zip(val_labels.columns, range(n_cols))), inplace=True)
    test_labels.rename(columns=dict(zip(test_labels.columns, range(n_cols))), inplace=True)
    bbox_files = set(list(img_index.keys()))
    bbox_patients = set([i.split('_')[0] for i in bbox_files])
    train_labels['patient'] = [i.split('_')[0] for i in train_labels.iloc[:, 0].values]
    val_labels['patient'] = [i.split('_')[0] for i in val_labels.iloc[:, 0].values]
    test_labels['patient'] = [i.split('_')[0] for i in test_labels.iloc[:, 0].values]

    # so split bbox patients into train and test sets.
    #
    # XXX need to preserve some kind of class balance tho. but that job is nearly
    # impossible because we have multiple disease subtypes per patient. So let's just
    # split first and see what shakes out.
    train_bbox_images = int(len(final_list) * TRAIN_TEST_SPLIT)
    bbox_patient_image_n = []
    bbox_patients = list(bbox_patients)
    for patient in bbox_patients:
        patient_len = 0
        for img in final_list:
            if patient == img[0].split('_')[0]:
                patient_len += 1
        bbox_patient_image_n.append(patient_len)

    bbox_train_patients = knapsack_splitter(bbox_patients, bbox_patient_image_n, train_bbox_images)
    bbox_test_patients = set(bbox_patients).difference(set(bbox_train_patients))

    # create new train and test sets
    tmp_train_set = train_labels.append(test_labels)
    tmp_train_set.index = range(len(tmp_train_set))

    new_test_img = tmp_train_set[tmp_train_set.patient.isin(bbox_test_patients)]
    new_train_set = tmp_train_set.loc[tmp_train_set.index.difference(new_test_img.index)]
    new_test_set = val_labels.append(new_test_img)

    # extract bbox items from both train and test sets
    train_bbox = new_train_set[new_train_set.iloc[:, 0].isin(bbox_files)]
    test_bbox = new_test_set[new_test_set.iloc[:, 0].isin(bbox_files)]

    new_train_set_nobbox = new_train_set.loc[new_train_set.index.difference(train_bbox.index)]
    new_test_set_nobbox = new_test_set.loc[new_test_set.index.difference(test_bbox.index)]
    new_train_set_nobbox = new_train_set_nobbox.drop(['patient'], axis=1)
    new_test_set_nobbox = new_test_set_nobbox.drop(['patient'], axis=1)

    # Save files
    with open('bbox_train_nobbox.processed', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(new_train_set_nobbox.values)

    with open('bbox_test_nobbox.processed', 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(new_test_set_nobbox.values)

    add_bbox_annos_to_bbox_set(train_bbox, final_list)
    add_bbox_annos_to_bbox_set(test_bbox, final_list)
    train_bbox = train_bbox.drop(['patient'], axis=1)
    test_bbox = test_bbox.drop(['patient'], axis=1)

    with open('bbox_train_withbbox.processed', 'w') as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(train_bbox.values)

    with open('bbox_test_withbbox.processed', 'w') as f:
        writer = csv.writer(f, delimiter=" ")
        writer.writerows(test_bbox.values)


if __name__ == "__main__":
    main()
