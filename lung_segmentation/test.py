import importAndProcess as iap
import model
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
dataset = iap.lungSegmentDataset("/media/minh/UStorage/NLM-MontgomeryCXRSet/MontgomerySet/CXR_png/", "/media/minh/UStorage/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/leftMask/", "/media/minh/UStorage/NLM-MontgomeryCXRSet/MontgomerySet/ManualMask/rightMask/",
imagetransform=Compose([Resize((400,400)),ToTensor(),normalize]),
labeltransform=Compose([Resize((400,400)),ToTensor()]),)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)

segNet = model.segmentNetwork().cuda()
segNet = torch.nn.DataParallel(segNet)
segNet.load_state_dict(torch.load('./650'))
show = iap.visualize(dataset)

for i,sample in enumerate(dataloader):
    img = torch.autograd.Variable(sample['image']).cuda()
    mask = segNet(img)
    show.ImageWithGround(i,True,True,save=True)
    show.ImageWithMask(i,mask.cpu().detach().numpy()[0],True,True,save=True)

