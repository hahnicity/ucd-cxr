import argparse
import csv
import os

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_entry_file")
    parser.add_argument("image_list")
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
