import torch.nn as nn
import torch.nn.functional as F
import math
import torch
class AttentionGate(nn.Module):
    def __init__(self, feature_channels, gate_channels, hidden_channels):
        super(AttentionGate, self).__init__()
        self.Wx = nn.Conv2d(
            in_channels=feature_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False)

        self.Wg = nn.Conv2d(
            in_channels=gate_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.psi = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)


    def forward(self,feature,attention):
        x = self.Wx(feature)
        g = F.upsample(self.Wg(attention),mode='bilinear',size=x.size()[2:])
        q_att = F.sigmoid(self.psi(F.relu(x + g)))
        alpha = F.upsample(q_att,mode='bilinear',size=feature.size()[2:])
        output = alpha.expand_as(feature) * feature
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class GuanResNet50_AG(torch.nn.Module):
    def __init__(self,block = Bottleneck, num_classes=14):
        self.inplanes = 64
        super(GuanResNet50_AG, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 3)
        self.layer2 = self._make_layer(block, 128, 4, stride=2)
        self.layer3 = self._make_layer(block, 256, 6, stride=2)
        self.layer4 = self._make_layer(block, 512, 3, stride=2)
        self.attention = AttentionGate(1024,2048,16).cuda()
        self.fc  = nn.Linear(3072,14,bias=True).cuda()
        self.sig = torch.nn.Sigmoid()
        self.avgpoolx4  = nn.AvgPool2d(7, stride=1)
        self.avgpoolag4 = nn.AvgPool2d(14, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            # backwards compatibility for serialized parameters
            if "fc" in name:
                 continue
            print("Loading")
            print(name)
            param = param.data
            own_state[name].copy_(param)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x  = self.layer1(x)
        x  = self.layer2(x)
        x  = self.layer3(x)
        x3 = x
        x  = self.layer4(x)
        x4 = x
        x  = self.avgpoolx4(x)
        ag = self.avgpoolag(self.attention(x3,x4))

        aggre1 = x.view(batch_size,-1)
        aggre2 = ag.view(batch_size,-1)
        aggre  = torch.cat((aggre1,aggre2),1)

        output = self.sig(self.fc(aggre))
        #print(output.size())

        return output

