# Copyright 2020 Huawei Technologies Co., Ltd
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
import os
from collections import OrderedDict
import mindspore.nn as nn
import mindspore.ops.operations as F
from mindspore.common.initializer import Normal
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import ParameterTuple

BN_MOMENTUM = 0.1


class MaxPool2dPytorch(nn.Cell):
    def __init__(self, kernel_size=1, stride=1, pad_mode="valid"):
        super(MaxPool2dPytorch, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.reverse = F.ReverseV2(axis=[2, 3])

    def construct(self, x):
        x = self.reverse(x)
        x = self.maxpool(x)
        x = self.reverse(x)
        return x


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.down_sample_layer = downsample
        self.stride = stride

    def construct(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample_layer is not None:
            residual = self.down_sample_layer(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Cell):

    def __init__(self, block, layers, cfg, pytorch_mode=True):
        self.inplanes = 64
        extra = cfg.MODEL.EXTRA
        self.deconv_with_bias = extra.DECONV_WITH_BIAS

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               pad_mode='pad', padding=3, has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        if pytorch_mode:
            self.maxpool = MaxPool2dPytorch(kernel_size=3, stride=2, pad_mode='same')
            print("use pytorch-style maxpool")
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
            print("use mindspore-style maxpool")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )

        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            pad_mode='pad',
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0,
            has_bias=True,
            weight_init=Normal(0.001),
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(OrderedDict([
                ('0', nn.Conv2d(self.inplanes, planes * block.expansion,
                                kernel_size=1, stride=stride, has_bias=False)),
                ('1', nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)),
            ]))

        layers = OrderedDict()
        layers['0'] = block(self.inplanes, planes, stride, downsample)
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers['{}'.format(i)] = block(self.inplanes, planes)

        return nn.SequentialCell(layers)

    def _get_deconv_cfg(self, deconv_kernel):
        assert deconv_kernel == 4, 'only support kernel_size = 4 for deconvolution layers'
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = OrderedDict()
        for i in range(num_layers):
            kernel, padding, _ = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers['deconv_{}'.format(i)] = nn.SequentialCell(OrderedDict([
                ('deconv', nn.Conv2dTranspose(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    pad_mode='pad',
                    padding=padding,
                    has_bias=self.deconv_with_bias,
                    weight_init=Normal(0.001),
                )),
                ('bn', nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)),
                ('relu', nn.ReLU()),
            ]))
            self.inplanes = planes

        return nn.SequentialCell(layers)

    def construct(self, x):
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
            weight = ParameterTuple(self.trainable_params())
            for w in weight:
                if w.name.split('.')[0] not in ('deconv_layers', 'final_layer'):
                    assert w.name in param_dict, "parameter %s not in checkpoint" % w.name
            load_param_into_net(self, param_dict)
            print('loading pretrained model {}'.format(pretrained))
        else:
            assert False, '{} is not a file'.format(pretrained)


resnet_spec = {50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(cfg, is_train, ckpt_path=None, pytorch_mode=False):
    num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    block_class, layers = resnet_spec[num_layers]
    model = PoseResNet(block_class, layers, cfg, pytorch_mode=pytorch_mode)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(ckpt_path)

    return model
