import argparse
from collections import OrderedDict
import csv
import os
import re

import numpy as np


position_mapping = {
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}

def knapsack_splitter(patients, sizes, capacity):
    """
    Split the patients and images into an optimally split validation set. We
    need knapsack because each patient is essentially an item in the knapsack
    and their value is the number of images they have

    patients: list of patients we are using
    sizes: # of images per patient
    capacity: # of images we ideally want in a dataset
    """
    patients = np.append([""], patients)
    sizes = np.append([0], sizes)
    # A boundary to just speed things up a bit. In our case we probably won't have
    # trouble meeting optimal capacity because most patients only have 1 img. But
    # it's helpful to have the code around in case.
    thresh = capacity * 0

    n = len(patients)
    patient_lookup = np.zeros((n+1, capacity+1))

    end_kp = False

    def reconstruct(i, w):
        """
        reconstruct subset of items i with capacity w. The two inputs
        i and w are taken at the point of optimality in the ip knapsack soln
        """
        recon = set()
        for ii in range(0, i+1)[::-1]:
            n_images_cur = patient_lookup[ii][w]
            n_images_prev = patient_lookup[ii-1][w]
            if n_images_cur > n_images_prev:
                recon.add(ii)
                w = w - sizes[ii]
        return recon

    max_kp = 0
    max_i = 0
    max_j = 0
    for i in range(1, n):
        for j in range(capacity+1):
            size = sizes[i]

            if size > j:
                patient_lookup[i][j] = patient_lookup[i-1][j]
            else:
                n_images_prev = patient_lookup[i-1][j]
                n_images = patient_lookup[i-1][j-size] + size
                if n_images > n_images_prev:
                    patient_lookup[i][j] = n_images
                    if (n_images >= capacity - thresh):
                        end_kp = True
                        break
                    elif (n_images > max_kp):
                        max_kp = n_images
                        max_i = i
                        max_j = j
                else:
                    patient_lookup[i][j] = patient_lookup[i-1][j]
        if end_kp:
            break

    if not end_kp:
        i = max_i
        j = max_j

    recon = reconstruct(i, j)
    return set(patients[list(recon)])


def perform_train_test_preprocessing(args):
    tmp_output = "{}.processed".format(os.path.splitext(args.image_list)[0])
    with open(tmp_output, 'w') as labeled_file:
        writer = csv.writer(labeled_file, delimiter=" ")
        image_list = sorted(map(lambda x: x.strip(), open(args.image_list).readlines()))
        de_idx = 0
        entry_reader = csv.reader(open(args.data_entry_file))
        data_entry = [line for line in entry_reader][1:]
        data_entry = sorted(data_entry, key=lambda x: x[0])
        for im_idx, image in enumerate(image_list):
            new_line = [image] + ([0] * 14)
            de_to_iter = data_entry[de_idx:]
            for entry in de_to_iter:
                de_idx += 1
                if entry[0].strip() != image:
                    continue
                findings = entry[1]
                if findings == 'No Finding':
                    break
                findings = findings.split('|')
                for finding in findings:
                    new_line[position_mapping[finding] + 1] = 1
                break
            writer.writerow(new_line)


def perform_validation_set_creation(args):
    train_set = np.genfromtxt(args.train_list, dtype='str')
    test_set = np.genfromtxt(args.test_list, dtype='str')
    valid_count = int((len(train_set) + len(test_set)) * (args.val_percent / 100))
    imgs = train_set[:, 0]
    train_pt_to_imgs = OrderedDict()
    for img in imgs:
        pt = re.search(r'(\d+)_', img).groups()[0]
        if pt not in train_pt_to_imgs:
            train_pt_to_imgs[pt] = 1
        else:
            train_pt_to_imgs[pt] += 1

    valid_pts = knapsack_splitter(list(train_pt_to_imgs.keys()), list(train_pt_to_imgs.values()), valid_count)
    train_rows_to_remove = []
    for idx, row in enumerate(train_set):
        pt = re.search(r'(\d+)_', row[0]).groups()[0]
        if pt in valid_pts:
            train_rows_to_remove.append(idx)
    valid_set = train_set[train_rows_to_remove]
    train_rows = list(set(range(len(train_set))).difference(set(train_rows_to_remove)))
    train_set = train_set[train_rows]

    label_dir = os.path.dirname(args.train_list)
    train_path = os.path.join(label_dir, "train_list.processed")
    valid_path = os.path.join(label_dir, "val_list.processed")
    np.savetxt(train_path, train_set, fmt='%s ')
    np.savetxt(valid_path, valid_set, fmt='%s ')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    initial_parser = subparsers.add_parser('train_test_preproc')
    initial_parser.add_argument("data_entry_file")
    initial_parser.add_argument("image_list")
    initial_parser.set_defaults(func=perform_train_test_preprocessing)
    valid_parser = subparsers.add_parser('make_validation_set')
    valid_parser.add_argument('train_list')
    valid_parser.add_argument('test_list')
    valid_parser.add_argument('--val-percent', help='overall percentage of total dataset validation set will use', type=int, default=10)
    valid_parser.set_defaults(func=perform_validation_set_creation)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
