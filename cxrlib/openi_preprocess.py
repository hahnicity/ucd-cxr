"""
openi_preprocess
~~~~~~~~~~~~~~~~

Preprocess openi dataset so that we can learn something from it. OpenI itself
is a semi-large dataset of CXRs that have an extraordinary amount of heterogenous
disease. We have over 120 possible pathologic findings from manual labeling alone.
Our goal is to hopefully reduce the number of things that our model has to predict.
By that we should probably only take the top number of items
"""
import argparse
import csv
from glob import glob
import operator
import os
import xml.etree.ElementTree as ET

from sklearn.model_selection import train_test_split


# XXX it would be good to talk to doctors about this
TAGS_TO_KEEP = {
    'Pulmonary Atelectasis': 0,
    'Cardiomegaly': 1,
    'Pleural Effusion': 2,
    'Infiltrate': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
    'Hydropneumothorax': 7,
    'Consolidation': 8,
    'Pulmonary Edema': 9,
    'Emphysema': 10,
    'Bullous Emphysema': 10,
    'Subcutaneous  Emphysema': 10,
    'Subcutaneous Emphysema': 10,
    'Pulmonary Emphysema': 10,
    'Pulmonary Fibrosis': 11,
    'Thickening': 12,
    'Hernia, Hiatal': 13,
    'Hernia, Diaphragmatic': 13,
}
KEEP_EXTRA_N = 6

def determine_lung_issue(text):
    if text == '':
        return text
    elif text.split('/')[0] in ['hypoinflation', 'hyperdistention']:
        return text.split('/')[0]
    else:
        return determine_lung_issue('/'.join(text.split('/')[1:]))


def output_file(filepath, processed):
    with open(filepath, 'w') as f:
        writer = csv.writer(f, delimiter=' ')
        writer.writerows(processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--output-dir')
    parser.add_argument('--train-percent', type=int, choices=[70, 80, 100], default=100)
    parser.add_argument('--validation-percent', type=int, choices=[10, 0], default=0)
    args = parser.parse_args()
    if args.train_percent == 100 and args.validation_percent == 10:
        raise Exception('Cannot utilize all data for training and 10% for validation too!')

    manual_tags = {}
    automatic_tags = {}
    report_struct = []

    tags = set()
    mesh_tags = set()
    reporting_dir = os.path.join(args.dir, 'ecgen-radiology')
    images_dir = os.path.join(args.dir, 'images')
    report_files = glob(os.path.join(reporting_dir, "*.xml"))
    if not report_files:
        raise Exception("Found no report files! Check the pathing you specified")

    for filepath in report_files:
        tree = ET.parse(filepath)
        root = tree.getroot()
        report_struct.append({
            "filename": filepath,
            "manual_tags": set(),
            "automatic_tags": set(),
            "associated_images": [],
        })
        for child in root.getchildren():
            tags.add(child.tag)
            if child.tag == "MeSH":
                for elem in child.getchildren():
                    diagnosis = elem.text.split('/')[0]
                    if diagnosis == "normal":
                        report_struct[-1]['manual_tags'].add(diagnosis)
                        continue

                    if diagnosis == "Lung":
                        diagnosis = determine_lung_issue(elem.text)

                    if diagnosis == '':
                        continue

                    mesh_tags.add(elem.tag)
                    if elem.tag == "major":
                        report_struct[-1]['manual_tags'].add(diagnosis)
                        if diagnosis not in manual_tags:
                            manual_tags[diagnosis] = 1
                        else:
                            manual_tags[diagnosis] += 1
                    elif elem.tag == "automatic":
                        report_struct[-1]['automatic_tags'].add(diagnosis)
                        if diagnosis not in automatic_tags:
                            automatic_tags[diagnosis] = 1
                        else:
                            automatic_tags[diagnosis] += 1

            if child.tag == 'parentImage':
                report_struct[-1]['associated_images'].append(child.attrib['id'])

    top_tags = sorted(manual_tags.items(), key=operator.itemgetter(1))[::-1]
    top_tags_wo_keep = [tag[0] for tag in top_tags if tag[0] not in TAGS_TO_KEEP.keys()]
    use_these_tags = list(TAGS_TO_KEEP.keys()) + top_tags_wo_keep[:KEEP_EXTRA_N]

    y_indexer = TAGS_TO_KEEP
    last_tag_lab = 13
    for tag in use_these_tags:
        if tag not in y_indexer:
            last_tag_lab += 1
            y_indexer[tag] = last_tag_lab

    processed = []
    for report in report_struct:
        y = [0] * (last_tag_lab + 1)
        # XXX Usually the first image in the report is the front image.
        # gotta check and make sure tho.
        #
        # There are xray images missing for a given report as well.
        if len(report['associated_images']) != 0:
            frontal_img = report['associated_images'][0] + '.png'
            for i in report['manual_tags']:
                try:
                    y[y_indexer[i]] = 1
                except KeyError:
                    continue
            processed.append([frontal_img] + y)

    output_dir = args.dir if not args.output_dir else args.output_dir
    if args.train_percent == 100:
        filepath = os.path.join(output_dir, "openi_labels_all_images_processed.csv")
        output_file(filepath, processed)
    else:
        train_len = int(len(processed) * (args.train_percent / 100))
        val_len = int(len(processed) * (args.validation_percent / 100))
        _, __, y_train, y_test = train_test_split([None] * len(processed), processed, train_size=(train_len+val_len))
        if args.validation_percent > 0:
            _, __, y_train, y_val = train_test_split([None] * len(y_train), y_train, train_size=train_len)
            val_filepath = os.path.join(output_dir, "openi_labels_val_processed.csv")
            output_file(val_filepath, y_val)

        train_filepath = os.path.join(output_dir, "openi_labels_train_processed.csv")
        test_filepath = os.path.join(output_dir, "openi_labels_test_processed.csv")
        output_file(train_filepath, y_train)
        output_file(test_filepath, y_test)


if __name__ == "__main__":
    main()
