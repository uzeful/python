'''--------------------------------------------------------------
PSPNet version of ResNet: Intergraded Version, Successed Version
---------------------------------------------------------------'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Front Block
class Frontblock(nn.Module):

    def __init__(self, inplanes, planes, stride=2):
        super(Frontblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 2, momentum=0.95)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out
    
    
# End Block
class Endblock(nn.Module):

    def __init__(self, inplanes, planes, num_classes=21, stride=1, padding=1):
        super(Endblock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.dropout1 = nn.Dropout2d(.1)
        self.conv2 = nn.Conv2d(planes, num_classes, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        out = self.conv2(out)

        return out
    

# Basic ResBlock
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, padding=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.95)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, momentum=0.95)
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


# PSPNet Pooling Branch
class PSPDec(nn.Module):

    def __init__(self, in_features, out_features, downsize, upsize=60, mode='bilinear'):
        super(PSPDec, self).__init__()

        self.averpool = nn.AvgPool2d(downsize, stride=downsize)
        self.conv = nn.Conv2d(in_features, out_features, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_features, momentum=.95)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(size=upsize, mode=mode)

    def forward(self, x):
        out = self.averpool(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.upsample(out)
        
        return out


class PSPNet(nn.Module):

    def __init__(self, block, layers, num_classes=21, scale_factor=8, mode='bilinear'):
        #self.inplanes = 64   # Original Res101
        self.inplanes = 128   # PSPNet Res101
        super(PSPNet, self).__init__()
        
        self.conv1 = Frontblock(3, 64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], padding=2, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], padding=4, dilation=4)

        self.layer5a = PSPDec(2048, 512, 60)
        self.layer5b = PSPDec(2048, 512, 30)
        self.layer5c = PSPDec(2048, 512, 20)
        self.layer5d = PSPDec(2048, 512, 10)
        
        self.fusion = Endblock(2048*2, 512, num_classes=num_classes)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

        # Initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, padding=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, padding, dilation, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, padding=padding, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = torch.cat([
            out,
            self.layer5d(x),
            self.layer5c(x),
            self.layer5b(x),
            self.layer5a(x),
        ], 1)
        
        out = self.fusion(out)
        out = self.upsample(out)

        return out


def pspnet101(pretrained=False, num_classes=21, **kwargs):
    """Constructs a PSPNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = PSPNet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)
    if pretrained:
        #import pdb
        #pdb.set_trace()
        model.load_state_dict(torch.load('/data/models/pspnet101_VOC2012.pth'))
    
    return model
