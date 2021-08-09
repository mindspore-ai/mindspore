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
'''
simple_baselines network
'''
from __future__ import division
import os
import mindspore.nn as nn
import mindspore.common.initializer as init
import mindspore.ops.operations as F
from mindspore.train.serialization import load_checkpoint, load_param_into_net

BN_MOMENTUM = 0.1

class MPReverse(nn.Cell):
    '''
    MPReverse
    '''
    def __init__(self, kernel_size=1, stride=1, pad_mode="valid"):
        super(MPReverse, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.reverse = F.ReverseV2(axis=[2, 3])

    def construct(self, x):
        x = self.reverse(x)
        x = self.maxpool(x)
        x = self.reverse(x)
        return x

class Bottleneck(nn.Cell):
    '''
    model part of network
    '''
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, has_bias=False, pad_mode='pad')
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        '''
        construct
        '''
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


class PoseResNet(nn.Cell):
    '''
    PoseResNet
    '''

    def __init__(self, block, layers, cfg):
        self.inplanes = 64
        self.deconv_with_bias = cfg.NETWORK.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, has_bias=False, pad_mode='pad')
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.maxpool = MPReverse(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            cfg.NETWORK.NUM_DECONV_LAYERS,
            cfg.NETWORK.NUM_DECONV_FILTERS,
            cfg.NETWORK.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=cfg.NETWORK.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=cfg.NETWORK.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if cfg.NETWORK.FINAL_CONV_KERNEL == 3 else 0,
            pad_mode='pad',
            has_bias=True,
            weight_init=init.Normal(0.001)
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        '''
        _make_layer
        '''
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([nn.Conv2d(self.inplanes, planes * block.expansion,
                                                      kernel_size=1, stride=stride, has_bias=False),
                                            nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            print(i)

        return nn.SequentialCell(layers)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        '''
        _make_deconv_layer
        '''
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        layers = []
        for i in range(num_layers):
            kernel = num_kernels[i]
            padding = 1
            planes = num_filters[i]

            layers.append(nn.Conv2dTranspose(
                in_channels=self.inplanes,
                out_channels=planes,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                has_bias=self.deconv_with_bias,
                pad_mode='pad',
                weight_init=init.Normal(0.001)
            ))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU())
            self.inplanes = planes

        return nn.SequentialCell(layers)

    def construct(self, x):
        '''
        construct
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)
        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            # load params from pretrained
            param_dict = load_checkpoint(pretrained)
            load_param_into_net(self, param_dict)
            print('=> loading pretrained model {}'.format(pretrained))
        else:
            print('=> imagenet pretrained model dose not exist')
            raise ValueError('{} is not a file'.format(pretrained))


resnet_spec = {50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def GetPoseResNet(cfg):
    '''
    GetPoseResNet
    '''
    num_layers = cfg.NETWORK.NUM_LAYERS
    block_class, layers = resnet_spec[num_layers]
    network = PoseResNet(block_class, layers, cfg)

    if cfg.MODEL.IS_TRAINED and cfg.MODEL.INIT_WEIGHTS:
        pretrained = ''
        if cfg.MODELARTS.IS_MODEL_ARTS:
            pretrained = cfg.MODELARTS.CACHE_INPUT + cfg.MODEL.PRETRAINED
        else:
            pretrained = cfg.TRAIN.CKPT_PATH + cfg.MODEL.PRETRAINED
        network.init_weights(pretrained)
    return network
