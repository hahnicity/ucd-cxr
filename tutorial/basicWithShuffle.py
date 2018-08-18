from sklearn.metrics import roc_auc_score
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.optim import Adam
from torch.nn import BCELoss
import torch.utils.model_zoo as model_zoo
from cxrlib.models import guan_resnet_ag
from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs

cur = 0

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

log = open("loss.txt","w")
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
# Run modified version of resnet50 on GPU
model = guan_resnet_ag.GuanResNet50_AG().cuda()
model.load_my_state_dict(model_zoo.load_url(model_urls['resnet50']))
#model.load_state_dict(torch.load(str(cur)))
# Ensure that we can run the model on multiple GPUs
#model = torch.nn.DataParallel(model)
model.train()

n_eps = 60 - cur - 1
max_batches = 0
optimizer = Adam(model.parameters(), lr=0.0001)
criterion = BCELoss()
normalize = Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
# Utilize customized chest xray dataset loading mechanism. Add additional transforms. Downsize all
# images to 224 pixels, then convert to tensor, and finally normalize.
dataset = ChestXrayDataSet('/media/minh/UStorage/chestxray14/images',
                           '/media/minh/UStorage/chestxray14/labels/train_val_list.processed',
                           transform=Compose([Resize(224), ToTensor(), normalize]))
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
batch_n = 0
for ep in range(n_eps):
    for inp, target in loader:
        target = torch.autograd.Variable(target).cuda()
        inp = torch.autograd.Variable(inp).cuda()
        out = model(inp)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
       # print(loss)
        print(loss.cpu().detach().numpy().item())
        log.write(str(loss.cpu().detach().numpy().item()))
        log.write("\n")
       # Quit after <max_batches> otherwise we will run for too long
       #if batch_n > max_batches:
        #    break
        #batch_n += 1
    torch.save(model.state_dict(),str(ep+cur+1))
    #print("end epoch {}".format(ep))
