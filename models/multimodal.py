import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


_all__ = [
    'simple', 'resnet-image', 'resnet-of', 'resnet-audio', 'resnet-all'
]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3(in_planes, out_planes, stride=1):
    """1D convolution size 3 with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class Basic3DResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic3DResNetBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Basic2DResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic2DResNetBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Basic1DResNetBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic1DResNetBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Basic3DBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic3DBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Basic2DBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic2DBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class Basic1DBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic1DBlock, self).__init__()
        self.conv1 = conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Bottleneck2D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
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


class MultiModal(nn.Module):
    def __init__(self, 
                 visual_block, 
                 audio_block,
                 visual_layers, 
                 audio_layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=101, 
                 video_pretrained=None):
        super(MultiModal, self).__init__()
        self.inplanes = 64

        # Visual modality
        self.v_conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.v_bn1 = nn.BatchNorm3d(64)
        self.v_relu = nn.ReLU(inplace=True)
        self.v_maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.v_layer1 = self._make_visual_layer(visual_block, 64, visual_layers[0], shortcut_type)
        self.v_layer2 = self._make_visual_layer(
            visual_block, 128, visual_layers[1], shortcut_type, stride=2)
        self.v_layer3 = self._make_visual_layer(
            visual_block, 256, visual_layers[2], shortcut_type, stride=2)
        self.v_layer4 = self._make_visual_layer(
            visual_block, 512, visual_layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.v_avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        # TODO - figure out how to load the weights 
        if video_pretrained:
            pass

        self.inplanes = 64

        # Audio modality
        self.a_conv1 = nn.Conv1d(1, 64, kernel_size=2, stride=2, padding=3,
                               bias=False)
        self.a_bn1 = nn.BatchNorm1d(64)
        self.a_relu = nn.ReLU(inplace=True)
        self.a_maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.a_layer1 = self._make_audio_layer(audio_block, 64, audio_layers[0], shortcut_type)
        self.a_layer2 = self._make_audio_layer(audio_block, 128, audio_layers[1], shortcut_type, stride=2)
        self.a_layer3 = self._make_audio_layer(audio_block, 256, audio_layers[2], shortcut_type, stride=2)
        self.a_layer4 = self._make_audio_layer(audio_block, 512, audio_layers[3], shortcut_type, stride=2)
        self.a_avgpool = nn.AdaptiveAvgPool1d(1)

        # TODO - check this 


        self.c_conv1 = nn.Conv1d(2, 64, kernel_size=2, stride=2, padding=3, bias=False)
        self.c_bn1 = nn.BatchNorm1d(64)
        self.c_relu = nn.ReLU(inplace=True)
        self.c_maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)


        #self.fc = nn.Linear(512 * 5 * (visual_block.expansion + audio_block.expansion), num_classes)
        self.fc = nn.Linear(512 * 5, num_classes)

        # Initializations
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_visual_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_audio_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, video_x, audio_x):
        video_x = self.v_conv1(video_x)
        video_x = self.v_bn1(video_x)
        video_x = self.v_relu(video_x)
        video_x = self.v_maxpool(video_x)

        video_x = self.v_layer1(video_x)
        video_x = self.v_layer2(video_x)
        video_x = self.v_layer3(video_x)
        video_x = self.v_layer4(video_x)

        video_x = self.v_avgpool(video_x)
        video_x = video_x.view(video_x.size(0), -1)

        audio_x = self.a_conv1(audio_x)
        audio_x = self.a_bn1(audio_x)
        audio_x = self.a_relu(audio_x)
        audio_x = self.a_maxpool(audio_x)

        audio_x = self.a_layer1(audio_x)
        audio_x = self.a_layer2(audio_x)
        audio_x = self.a_layer3(audio_x)
        audio_x = self.a_layer4(audio_x)

        audio_x = self.a_avgpool(audio_x)
        audio_x = audio_x.view(audio_x.size(0), -1)

        x = torch.cat([video_x, audio_x], dim=1)

        x = self.fc(x)
        return x

def simple10(**kwargs):
    model = MultiModal(Basic3DBlock, Basic1DBlock, [1,1,1,1], [1,1,1,1], **kwargs)
    return model


def multimodal_rr_18(pretrained=False, **kwargs):
    """Constructs a MultiModal-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MultiModal(Basic3DResNetBlock, Basic2DResNetBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multimodal18']))
    return model


def multimodal34(pretrained=False, **kwargs):
    """Constructs a MultiModal-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MultiModal(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multimodal34']))
    return model


def multimodal50(pretrained=False, **kwargs):
    """Constructs a MultiModal-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MultiModal(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multimodal50']))
    return model


def multimodal101(pretrained=False, **kwargs):
    """Constructs a MultiModal-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MultiModal(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multimodal101']))
    return model


def multimodal152(pretrained=False, **kwargs):
    """Constructs a MultiModal-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MultiModal(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multimodal152']))
    return model

