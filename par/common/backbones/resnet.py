from torch import nn
from torchvision.models.resnet import (
    BasicBlock, Bottleneck, conv1x1, model_urls
)
from torchvision.models.utils import load_state_dict_from_url

from par.common.backbones.utils import load_state_dict


def resnet18(last_layer_stride=2, pretrained=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], last_layer_stride,
                   pretrained, **kwargs)


def resnet34(last_layer_stride=2, pretrained=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], last_layer_stride,
                   pretrained, **kwargs)


def resnet50(last_layer_stride=2, pretrained=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], last_layer_stride,
                   pretrained, **kwargs)


def resnet101(last_layer_stride=2, pretrained=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], last_layer_stride,
                   pretrained,
                   **kwargs)


def resnet152(last_layer_stride=2, pretrained=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], last_layer_stride,
                   pretrained,
                   **kwargs)


def resnext50_32x4d(last_layer_stride=2, pretrained=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   last_layer_stride, pretrained, **kwargs)


def resnext101_32x8d(last_layer_stride=2, pretrained=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   last_layer_stride, pretrained, **kwargs)


def wide_resnet50_2(last_layer_stride=2, pretrained=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   last_layer_stride, pretrained, **kwargs)


def wide_resnet101_2(last_layer_stride=2, pretrained=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   last_layer_stride, pretrained, **kwargs)


def _resnet(arch, block, layers, last_layer_stride, pretrained, **kwargs):
    model = ResNet(block, layers, last_layer_stride, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=True)
        load_state_dict(model, state_dict)

    return model, 512 * block.expansion


class ResNet(nn.Module):

    def __init__(self, block, layers, last_layer_stride=2,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or "
                f"a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, 7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=last_layer_stride,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups=self.groups,
                            base_width=self.base_width,
                            dilation=previous_dilation,
                            norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
