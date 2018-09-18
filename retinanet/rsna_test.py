import argparse
import csv
import multiprocessing

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.encoder import DataEncoder
from retinanet.datagen import ListDataset
from retinanet.retinanet_bbox import RetinaNet
from retinanet.transform import resize_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_file')
    parser.add_argument('test_image_dir')
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint_file)
    net = RetinaNet(num_classes=1).cuda()
    net.load_state_dict(ckpt['net'])
    input_size = 224

    test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        #transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])
    data = ListDataset(args.test_image_dir, None, False, test_transforms, input_size)
    loader = DataLoader(data, batch_size=16, num_workers=multiprocessing.cpu_count())
    encoder = DataEncoder()
    net.eval()
    with torch.no_grad():
        with open('kaggle-submission.txt', 'w') as sub_file:
            writer = csv.writer(sub_file)
            writer.writerow(['patientId', 'PredictionString'])
            for inputs, patients in loader:
                inputs = Variable(inputs, volatile=True).cuda()
                loc_preds, cls_preds = net(inputs)
                for i in range(len(loc_preds)):
                    boxes, labels = encoder.decode(loc_preds[i], cls_preds[i], input_size)
                    boxes = boxes.cpu()
                    if boxes.size(0) > 0:
                        boxes = boxes.squeeze(1)
                        boxes = resize_boxes(boxes, 224, 1024)
                    boxes = boxes.numpy()

                    boxes_out = []
                    for box_idx, box in enumerate(boxes):
                        boxes_out.append(labels[box_idx].cpu().numpy()[0])
                        boxes_out.extend([box[0], box[1]])
                        width = box[2] - box[0]
                        height = box[3] - box[1]
                        boxes_out.extend([width, height])
                    writer.writerow([patients[i]] + [" ".join([str(j) for j in boxes_out])])


if __name__ == "__main__":
    main()
