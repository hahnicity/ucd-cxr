from sklearn.metrics import roc_auc_score
import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torch.optim import SGD
from torch.nn import BCELoss

from cxrlib.models import guan_resnet_ag
from cxrlib.read_data import ChestXrayDataSet
from cxrlib.results import compute_AUCs

# Run modified version of resnet50 on GPU
model = guan_resnet_ag.GuanResNet50_AG().cuda()
# Ensure that we can run the model on multiple GPUs
#model = torch.nn.DataParallel(model)
model.train()

n_eps = 1
max_batches = 16
optimizer = SGD(model.parameters(), lr=0.01)
criterion = BCELoss()
normalize = Normalize([0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225])
# Utilize customized chest xray dataset loading mechanism. Add additional transforms. Downsize all
# images to 224 pixels, then convert to tensor, and finally normalize.
dataset = ChestXrayDataSet('/fastdata/chestxray14/images',
                           '/fastdata/chestxray14/labels/train_val_list.processed',
                           transform=Compose([Resize(224), ToTensor(), normalize]))
loader = torch.utils.data.DataLoader(dataset, batch_size=64)
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

        # Quit after <max_batches> otherwise we will run for too long
        if batch_n > max_batches:
            break
        batch_n += 1

    print("end epoch {}".format(ep))

# Load testing dataset
test_dataset = ChestXrayDataSet('/fastdata/chestxray14/images',
                                '/fastdata/chestxray14/labels/test_list.processed',
                                transform=Compose([Resize(224), ToTensor(), normalize]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

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
    import IPython; IPython.embed()
    # Add results of the model's output to the aggregated prediction vector, and also add aggregated
    # ground truth information as well
    pred = torch.cat((pred, out.data), 0)
    gt = torch.cat((gt, target.data), 0)

    if batch_n > max_batches:
        break
    batch_n += 1

# Compute the model area under curve (AUC).
auc = compute_AUCs(gt, pred)
print("AUC Results: {}".format(sum(auc) / len(auc)))
