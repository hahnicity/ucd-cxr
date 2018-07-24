import torch
import torch.nn as nn
from torch.utils import model_zoo
from torchvision.models import resnet50

from cxrlib.models.resnet_grayscale import Bottleneck

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}


class BottleNeckAttentionModule(torch.nn.Module):
    def __init__(self, input_size, input_chan, r, d):
        """
        Bottleneck Attention Module.

        :param input_size: size of input image. Must be in terms of NxN
        :param input_chan: number of input channels
        :param r: reduction ratio in channel attention branch
        :param d: dilation on spatial attention
        """
        super(BottleNeckAttentionModule, self).__init__()
        reduced = int(input_chan/r)
        self.spatial_conv1 = nn.Conv2d(input_chan, reduced, 1, bias=False)
        self.spatial_conv2 = nn.Conv2d(
            reduced, reduced, 3, padding=d, dilation=d, bias=False
        )
        self.spatial_conv3 = nn.Conv2d(
            reduced, reduced, 3, padding=d, dilation=d, bias=False
        )
        self.spatial_conv4 = nn.Conv2d(reduced, 1, 1, bias=False)
        self.global_av = nn.AvgPool2d(input_size)
        self.fc1 = nn.Linear(input_chan, reduced)
        self.fc2 = nn.Linear(reduced, input_chan)
        self.bn = nn.BatchNorm2d(input_chan)
        self.sig = nn.Sigmoid()

    def _channel_attention(self, x):
        x = self.global_av(x)
        # flatten vector
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.unsqueeze(2).unsqueeze(3)
        return self.bn(x)

    def _spatial_attention(self, x):
        x = self.spatial_conv1(x)
        x = self.spatial_conv2(x)
        x = self.spatial_conv3(x)
        return self.spatial_conv4(x)

    def forward(self, x):
        atten = self.sig(self._channel_attention(x) + self._spatial_attention(x))
        atten = torch.mul(atten, x)
        return atten + x


class BAMResNet(nn.Module):

    def __init__(self, resnet, r, d):
        expansion = 4
        super(BAMResNet, self).__init__()
        self.resnet = resnet
        self.bam1 = BottleNeckAttentionModule(56, 64*expansion, r, d)
        self.bam2 = BottleNeckAttentionModule(28, 128*expansion, r, d)
        self.bam3 = BottleNeckAttentionModule(14, 256*expansion, r, d)
        self.resnet.fc = nn.Linear(512 * expansion, 14)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.bam1(x)
        x = self.resnet.layer2(x)
        x = self.bam2(x)
        x = self.resnet.layer3(x)
        x = self.bam3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.resnet.fc(x)

        return x


def resnet50ish(r, d, **kwargs):
    """
    ResNet50 but with BAM.

    :param r: reduction ratio in channel attention branch
    :param d: dilation on spatial attention
    """
    # load resnet50
    model = resnet50()
    model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    # then initialize BAM on top of the pretrained resnet
    model = BAMResNet(model, r, d)
    return model
