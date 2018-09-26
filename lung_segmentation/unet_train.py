import argparse
import os

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import importAndProcess as iap
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from cxrlib.models.unet_models import unet11


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('montgomery_path')
    parser.add_argument('-r', '--resume-from', help='resume from a specific savepoint')
    parser.add_argument('-e', '--epochs', default=700, type=int)
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    args = parser.parse_args()

    log = open("log.txt","a")
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dataset = iap.lungSegmentDataset(
        os.path.join(args.montgomery_path, "CXR_png"),
        os.path.join(args.montgomery_path, "ManualMask/leftMask/"),
        os.path.join(args.montgomery_path, "ManualMask/rightMask/"),
        imagetransform=Compose([Resize((224, 224)),ToTensor(),normalize]),
        labeltransform=Compose([Resize((224, 224)),ToTensor()]),
        convert_to='RGB',
    )

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size, shuffle=True)
    segNet = unet11(pretrained=True, out_filters=3).cuda()
    segNet = torch.nn.DataParallel(segNet)
    if args.resume_from:
        segNet.load_state_dict(torch.load(args.resume_from))
        init_epochs = int(os.path.basename(args.resume_from).replace('unet11_', ''))
    else:
        init_epochs = 0
    optimizer = Adam(segNet.parameters(), lr=0.0002)
    criterion = CrossEntropyLoss()
    for ep_offset in range(args.epochs-init_epochs):
        eps = ep_offset + init_epochs
        for sample in dataloader:
            img = torch.autograd.Variable(sample['image']).cuda()
            ground = torch.autograd.Variable(sample['label']).long().cuda()
            # the output mask seems to have 1 channel, which is different than
            # the output mask for Sam's resnet which has 3 channels
            mask = segNet(img)
            optimizer.zero_grad()
            loss = criterion(mask,ground)
            loss.backward()
            optimizer.step()
            print(loss.cpu().detach().numpy().item())
            log.write(str(loss.cpu().detach().numpy().item()) + "\n")

        if((eps+1) % 50 == 0):
            torch.save(segNet.state_dict(),"unet11_" + str(eps+1))

if __name__ == '__main__':
    main()
