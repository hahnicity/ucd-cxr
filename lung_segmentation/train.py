from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import importAndProcess as iap
import model
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
def main():
    log = open("log.txt","a")
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dataset = iap.lungSegmentDataset("/media/minh/UStorage/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/", "/media/minh/UStorage/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/", "/media/minh/UStorage/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/",
    imagetransform=Compose([Resize((400,400)),ToTensor(),normalize]),
    labeltransform=Compose([Resize((400,400)),ToTensor()]),)

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=10,shuffle=True)
    segNet = model.segmentNetwork().cuda()
    segNet = torch.nn.DataParallel(segNet)
    segNet.load_state_dict(torch.load('./100'))
    optimizer = Adam(segNet.parameters(), lr=0.0002)
    criterion = CrossEntropyLoss()
    num_epochs = 700
    for eps in range(num_epochs):
        for sample in dataloader:
            img = torch.autograd.Variable(sample['image']).cuda()
            ground = torch.autograd.Variable(sample['label']).long().cuda()

            mask = segNet(img)
            optimizer.zero_grad()
            loss = criterion(mask,ground)
            loss.backward()
            optimizer.step()
            print(loss.cpu().detach().numpy().item())
            log.write(str(loss.cpu().detach().numpy().item()) + "\n")

        if(eps % 50 == 0):
            torch.save(segNet.state_dict(),str(eps))

if __name__ == '__main__':
    main()

