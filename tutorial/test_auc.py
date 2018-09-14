from sklearn.metrics import roc_auc_score
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.optim import SGD
from torch.nn import BCELoss
import torch.utils.model_zoo as model_zoo
from cxrlib.models import guan_resnet_ag
from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs

cur = 17

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

log = open("auc.txt","a")
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
model = guan_resnet_ag.GuanResNet50_AG().cuda()
model.load_state_dict(torch.load(str(cur)))
model = torch.nn.DataParallel(model)
# Ensure that we can run the model on multiple GPUs



# Initialize two empty vectors that we can use in the future for storing aggregated ground truth (gt)
# and model prediction (pred) information.
batch_n = 0
normalize = Normalize([0.485, 0.456, 0.406],
                  [0.229, 0.224, 0.225])
test_dataset = ChestXrayDataSet('/media/minh/UStorage/chestxray14/images',
                            '/media/minh/UStorage/chestxray14/labels/test_list.processed',
                            transform=Compose([Resize(224), ToTensor(), normalize]))
loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
gt = torch.FloatTensor().cuda()
pred = torch.FloatTensor().cuda() 
model.eval()

for i in range(1,2):
    for inp, target in loader:
        target = torch.autograd.Variable(target).cuda()
        inp = torch.autograd.Variable(inp).cuda()
        out = model(inp)
        # Add results of the model's output to the aggregated prediction vector, and also add aggregated
        # ground truth information as well
        pred = torch.cat((pred, out.data), 0)
        gt = torch.cat((gt, target.data), 0)
        print(batch_n)
        #if batch_n > max_batches:
        #    break
        batch_n += 1

    # Compute the model area under curve (AUC).
    auc = compute_AUCs(gt, pred)
    print("AUC Results: {}".format(sum(auc) / len(auc)))
    
    log.write(str(sum(auc)/len(auc)))
    log.write("\n")
    log.write(str(cur))
    log.write("============")
    del test_dataset
    del loader
    del model
    del gt
    del pred

