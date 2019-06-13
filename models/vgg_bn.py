"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from layers.gate_layer import GateLayer

__all__ = [
    'slimmingvgg',
]

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}

class LinView(nn.Module):
    def __init__(self):
        super(LinView, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VGG(nn.Module):

    def __init__(self, features, cfg, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(cfg[0] * 7 * 7, cfg[1]),
            nn.BatchNorm1d(cfg[1]),
            nn.ReLU(True),
            nn.Linear(cfg[1],cfg[2]),
            nn.BatchNorm1d(cfg[2]),
            nn.ReLU(True),
            nn.Linear(cfg[2], num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')#, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M', 4096, 4096]
}


def flatten_model(old_net):
    """Removes nested modules. Only works for VGG."""
    from collections import OrderedDict
    module_list, counter, inserted_view = [], 0, False
    gate_counter = 0
    print("printing network")
    print(" Hard codded network in vgg_bn.py")
    for m_indx, module in enumerate(old_net.modules()):
        if not isinstance(module, (nn.Sequential, VGG)):
            print(m_indx, module)
            if isinstance(module, nn.Linear) and not inserted_view:
                module_list.append(('flatten', LinView()))
                inserted_view = True

            # features.0
            # classifier
            prefix = "features"

            if m_indx > 30:
                prefix = "classifier"
            if m_indx == 32:
                counter = 0

            # prefix = ""

            module_list.append((prefix + str(counter), module))

            if isinstance(module, nn.BatchNorm2d):
                planes = module.num_features
                gate = GateLayer(planes, planes, [1, -1, 1, 1])
                module_list.append(('gate%d' % (gate_counter), gate))
                print("gate ", counter, planes)
                gate_counter += 1


            if isinstance(module, nn.BatchNorm1d):
                planes = module.num_features
                gate = GateLayer(planes, planes, [1, -1])
                module_list.append(('gate%d' % (gate_counter), gate))
                print("gate ", counter, planes)
                gate_counter += 1


            counter += 1
    new_net = nn.Sequential(OrderedDict(module_list))
    return new_net


def slimmingvgg(pretrained=False, config=None, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    if config == None:
        config = cfg['A']
    config2 = [config[-4],config[-2],config[-1]]
    model = VGG(make_layers(config[:-2], batch_norm=True), config2, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']),strict=False)
    model = flatten_model(model)
    return model
