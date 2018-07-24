import torch
import torch.nn as nn

from cxrlib.models.resnet_grayscale import Bottleneck


class BottleNeckAttentionModule(torch.nn.Module):
    def __init__(self, input_size, input_chan, r, d):
        """
        :param input_size: size of input image. Must be in terms of NxN
        :param input_chan: number of input channels
        :param r: reduction ratio in channel attention branch
        :param d: dilation on spatial attention
        """
        reduced = int(input_chan/r)
        self.spatial_conv1 = nn.Conv2d(input_chan, reduced, 1, bias=False)
        # XXX need to figure out padding
        self.spatial_conv2 = nn.Conv2d(reduced, reduced, 3, dilation=d, bias=False)
        self.spatial_conv3 = nn.Conv2d(reduced, reduced, 3, dilation=d, bias=False)
        self.spatial_conv4 = nn.Conv2d(reduced, 1, 1, bias=False)
        self.global_av = nn.AvgPool2d(input_size)
        self.fc1 = nn.Linear(input_chan, reduced)
        self.fc2 = nn.Linear(reduced, input_chan)
        self.bn = nn.BatchNorm2d(input_chan)

    def _channel_attention(self, x):
        x = self.global_av(x)
        # flatten vector
        x = x.view(x.size(0), -1)
        return self.bn(self.fc2(self.fc1(x)))

    def _spatial_attention(self, x):
        x = self.spatial_conv1(x)
        x = self.spatial_conv2(x)
        x = self.spatial_conv3(x)
        return self.spatial_conv4(x)

    def forward(self, x):
        atten = nn.Sigmoid(self._channel_attention(x) + self._spatial_attention(x))
        atten = torch.mul(atten, x)
        return atten + x


class BAMResNet(nn.Module):

    def __init__(self, block, layers, r, d, num_classes=14):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bam1 = self.BottleNeckAttentionModule(128, 64*block.expansion, r, d)
        self.bam2 = self.BottleNeckAttentionModule(64, 128*block.expansion, r, d)
        self.bam3 = self.BottleNeckAttentionModule(32, 256*block.expansion, r, d)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.bam1(x)
        x = self.layer2(x)
        x = self.bam2(x)
        x = self.layer3(x)
        x = self.bam3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50ish(r, d, **kwargs):
    """
    :param r: reduction ratio in channel attention branch
    :param d: dilation on spatial attention
    """
    model = BAMResNet(Bottleneck, [3, 4, 6, 3], r, d)
    return model
