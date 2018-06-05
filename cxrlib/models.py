import torch
import torchvision


class GuanResNet50(torch.nn.Module):
    def __init__(self):
        super(GuanResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        return self.sigmoid(x)
