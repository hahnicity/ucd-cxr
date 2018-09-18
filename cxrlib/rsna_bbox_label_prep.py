"""
rsna_bbox_label_prep
~~~~~~~~~~~~~~~~~~~~

For retinanet at least, the desired output is:

    img xmin1 ymin1 xmax1 ymax1 cls_idx1 xmin2 ...

It can just be done in csv
"""
import argparse
import csv

from numpy.random import choice


def process_file(train_input_filepath, train_output_filepath, validation_output_filepath, validation_frac):
    patient_dict = {}
    with open(train_input_filepath, "r") as f:
        reader = csv.reader(f)
        for idx, l in enumerate(reader):
            if idx == 0:  # skip header
                continue
            patient, x, y, w, h, cls = l
            if cls == '0':
                x_min, y_min, y_max, x_max = [0] * 4
                # I think retinanet ignores things < 0
                cls = -2
            else:
                x_min = float(x)
                y_min = float(y)
                x_max = x_min + float(w)
                y_max = y_min + float(h)
                cls = 0

            if y_max > 1024:
                print('you got a x problem: line {}'.format(idx))

            if patient in patient_dict:
                patient_dict[patient].extend([x_min, y_min, x_max, y_max, cls])
            else:
                patient_dict[patient] = [x_min, y_min, x_max, y_max, cls]

    # randomly choose patients to be in validation set
    train_n = len(patient_dict)
    validation_n = int(train_n * validation_frac)
    validation_patients = choice(list(patient_dict.keys()), size=validation_n)

    with open(train_output_filepath, 'w') as f_train:
        with open(validation_output_filepath, 'w') as f_val:
            writer_train = csv.writer(f_train)
            writer_val = csv.writer(f_val)
            for patient in patient_dict:
                if patient not in validation_patients:
                    writer_train.writerow(["{}.dcm".format(patient)] + patient_dict[patient])
                else:
                    writer_val.writerow(["{}.dcm".format(patient)] + patient_dict[patient])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_input_filepath')
    parser.add_argument('train_output_filepath')
    parser.add_argument('validation_output_filepath')
    parser.add_argument('--validation-frac', type=float, default=.2)
    args = parser.parse_args()
    process_file(args.train_input_filepath, args.train_output_filepath, args.validation_output_filepath, args.validation_frac)


if __name__ == "__main__":
    main()
