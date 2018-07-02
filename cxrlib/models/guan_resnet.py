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
    def __init__(self, pretrained=True):
        super(GuanResNet50, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained, num_classes=14)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        return self.sig(out)


class GuanResNet50Grayscale(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(GuanResNet50Grayscale, self).__init__()
        self.model = torchvision.models.resnet50(pretrained=pretrained, num_classes=14)
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.sig = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        return self.sig(out)
