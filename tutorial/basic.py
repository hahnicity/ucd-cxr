from sklearn.metrics import roc_auc_score
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.optim import SGD
from torch.nn import BCELoss
import torch.utils.model_zoo as model_zoo
from cxrlib.models import guan_resnet_ag
from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs

cur = -1

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

# Ensure that we can run the model on multiple GPUs
model = torch.nn.DataParallel(model)
#model.load_state_dict(torch.load(str(cur)))
model.train()

n_eps = 60 - cur - 1
max_batches = 16
optimizer = SGD(model.parameters(), lr=0.001)
criterion = BCELoss()
normalize = Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
# Utilize customized chest xray dataset loading mechanism. Add additional transforms. Downsize all
# images to 224 pixels, then convert to tensor, and finally normalize.
dataset = ChestXrayDataSet('/fastdata/chestxray14/images',
                           '/fastdata/chestxray14/labels/train_val_list.processed',
                           transform=Compose([Resize(224), ToTensor(), normalize]))
loader = torch.utils.data.DataLoader(dataset, batch_size=32)
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
       # print(loss.cpu().detach().numpy().item())
        log.write(str(loss.cpu().detach().numpy().item()))
        log.write("\n")
       # Quit after <max_batches> otherwise we will run for too long
       #if batch_n > max_batches:
        #    break
        #batch_n += 1
    torch.save(model.state_dict(),str(ep+cur+1))
    #print("end epoch {}".format(ep))
# Load testing dataset
test_dataset = ChestXrayDataSet('/fastdata/chestxray14/images',
                                '/fastdata/chestxray14/labels/test_list.processed',
                                transform=Compose([Resize(224), ToTensor(), normalize]))
loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

# Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
# and model prediction (pred) information.
gt = torch.FloatTensor().cuda()
pred = torch.FloatTensor().cuda()
model.eval()

batch_n = 0
for inp, target in loader:
    target = torch.autograd.Variable(target).cuda()
    inp = torch.autograd.Variable(inp).cuda()
    out = model(inp)
    # Add results of the model's output to the aggregated prediction vector, and also add aggregated
    # ground truth information as well
    pred = torch.cat((pred, out.data), 0)
    gt = torch.cat((gt, target.data), 0)
    print("end batch")
    #if batch_n > max_batches:
    #    break
    #batch_n += 1

# Compute the model area under curve (AUC).
auc = compute_AUCs(gt, pred)
print("AUC Results: {}".format(sum(auc) / len(auc)))
