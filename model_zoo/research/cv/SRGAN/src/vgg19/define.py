# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Structure of VGG19"""

import mindspore.nn as nn
from mindspore.train.serialization import load_checkpoint, load_param_into_net

class VGG(nn.Cell):
    """Structure of VGG19"""

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features

    def construct(self, x):
        x = self.features(x)
        return x

def make_layers(mycfg, batch_norm=False):
    """make network"""
    layers = []
    in_channels = 3
    for v in mycfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3,
                               pad_mode='pad', padding=1, has_bias=True)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(num_features=v, momentum=0.9), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell([*layers])

cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def vgg19(vgg_ckpt):
    """VGG 19-layer model (configuration "19")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['19']))
    param_dict = load_checkpoint(vgg_ckpt)
    load_param_into_net(model, param_dict)
    return model
