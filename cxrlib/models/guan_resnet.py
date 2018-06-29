"""
guan_resnet
~~~~~~~~~~~

Guan's global branch. The only intention was to prove true results with no
overlapping patients would be much lower than reported results with overlapping
patients in train and test sets
"""
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
