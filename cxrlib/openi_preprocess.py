import argparse
from glob import glob
import os
import xml.etree.ElementTree as ET


def determine_lung_issue(text):
    if text == '':
        return text
    elif text.split('/')[0] in ['hypoinflation', 'hyperdistention']:
        return text.split('/')[0]
    else:
        return determine_lung_issue('/'.join(text.split('/')[1:]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    args = parser.parse_args()
    manual_tags = {}
    automatic_tags = {}
    report_struct = []

    tags = set()
    mesh_tags = set()

    for filepath in glob(os.path.join(args.dir, "*.xml")):
        tree = ET.parse(filepath)
        root = tree.getroot()
        report_struct.append({
            "filename": filepath,
            "tags": set(),
        })
        for child in root.getchildren():
            tags.add(child.tag)
            if child.tag == "MeSH":
                for elem in child.getchildren():
                    diagnosis = elem.text.split('/')[0]
                    if diagnosis == "normal":
                        report_struct[-1]['tags'].add(diagnosis)
                        continue

                    if diagnosis == "Lung":
                        diagnosis = determine_lung_issue(elem.text)
                    mesh_tags.add(elem.tag)
                    if elem.tag == "major":
                        report_struct[-1]['tags'].add(diagnosis)
                        if diagnosis not in manual_tags:
                            manual_tags[diagnosis] = 1
                        else:
                            manual_tags[diagnosis] += 1
                    elif elem.tag == "automatic":
                        if diagnosis not in automatic_tags:
                            automatic_tags[diagnosis] = 1
                        else:
                            automatic_tags[diagnosis] += 1

    import IPython; IPython.embed()


if __name__ == "__main__":
    main()
