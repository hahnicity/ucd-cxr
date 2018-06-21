import torch
import torchvision

class GuanResNet50(torch.nn.Module):
    def __init__(self):
        super(GuanResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        return self.sig(out)

checkpoint = torch.load('global_weights_2018_06_06_2320.pt')
model = GuanResNet50()
model = torch.nn.DataParallel(model)
model.load_state_dict(checkpoint)
