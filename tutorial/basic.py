import torch
from torchvision.models import resnet50
from torch.optim import SGD
from torch.nn import MSELoss

from cxrlib.read_data import RandomDataset

model = resnet50().cpu()
model.fc = torch.nn.Linear(2048, 1)

n_eps = 5
optimizer = SGD(model.parameters(), lr=0.01)
criterion = MSELoss()
dataset = RandomDataset(transform=Compose([normalize]))

for ep in range(n_eps):
    for inp, target in data:
        target = torch.autograd.Variable(target)
        inp = torch.autograd.Variable(inp)
        out = model(inp)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
    print("end epoch {}".format(ep))
