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
"""Network."""
import math
from functools import reduce
import numpy as np

import mindspore
import mindspore.nn as nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore import context, Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

# ResNet
def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)

    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=1, pad_mode='pad', weight_init=weight)

def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)

    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='pad', weight_init=weight)

def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)

    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight)

def _bn(channel):
    return nn.BatchNorm2d(channel)


def _bn_last(channel):
    return nn.BatchNorm2d(channel)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)

class ResidualBlock(nn.Cell):
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()

        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last(out_channel)

        self.relu = nn.ReLU()

        self.down_sample = False

        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell([_conv1x1(in_channel, out_channel, stride),
                                                        _bn(out_channel)])
        self.add = P.Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out

class ResNet(nn.Cell):
    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes):
        super(ResNet, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = _conv7x7(3, 64, stride=2)
        self.bn1 = _bn(64)
        self.relu = P.ReLU()


        self.pad = P.Pad(((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="valid")


        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2])
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])

        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []

        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pad(x)

        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return c3, c4, c5

def resnet50(class_num=10):
    return ResNet(ResidualBlock,
                  [3, 4, 6, 3],
                  [64, 256, 512, 1024],
                  [256, 512, 1024, 2048],
                  [1, 2, 2, 2],
                  class_num)


# RetinaFace
def Init_KaimingUniform(arr_shape, a=0, nonlinearity='leaky_relu', has_bias=False):
    def _calculate_in_and_out(arr_shape):
        dim = len(arr_shape)
        if dim < 2:
            raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

        n_in = arr_shape[1]
        n_out = arr_shape[0]

        if dim > 2:

            counter = reduce(lambda x, y: x * y, arr_shape[2:])
            n_in *= counter
            n_out *= counter
        return n_in, n_out

    def calculate_gain(nonlinearity, a=None):
        linear_fans = ['linear', 'conv1d', 'conv2d', 'conv3d',
                       'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
        if nonlinearity in linear_fans or nonlinearity == 'sigmoid':
            return 1
        if nonlinearity == 'tanh':
            return 5.0 / 3
        if nonlinearity == 'relu':
            return math.sqrt(2.0)
        if nonlinearity == 'leaky_relu':
            if a is None:
                negative_slope = 0.01
            elif not isinstance(a, bool) and isinstance(a, int) or isinstance(a, float):
                negative_slope = a
            else:
                raise ValueError("negative_slope {} not a valid number".format(a))
            return math.sqrt(2.0 / (1 + negative_slope ** 2))

        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    fan_in, _ = _calculate_in_and_out(arr_shape)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    weight = np.random.uniform(-bound, bound, arr_shape).astype(np.float32)

    bias = None
    if has_bias:
        bound_bias = 1 / math.sqrt(fan_in)
        bias = np.random.uniform(-bound_bias, bound_bias, arr_shape[0:1]).astype(np.float32)
        bias = Tensor(bias)

    return Tensor(weight), bias

class ConvBNReLU(nn.SequentialCell):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer, leaky=0):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = Init_KaimingUniform(weight_shape, a=math.sqrt(5))

        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
            nn.LeakyReLU(alpha=leaky)
        )

class ConvBN(nn.SequentialCell):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, groups, norm_layer):
        weight_shape = (out_planes, in_planes, kernel_size, kernel_size)
        kaiming_weight, _ = Init_KaimingUniform(weight_shape, a=math.sqrt(5))

        super(ConvBN, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', padding=padding, group=groups,
                      has_bias=False, weight_init=kaiming_weight),
            norm_layer(out_planes),
        )

class SSH(nn.Cell):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1

        norm_layer = nn.BatchNorm2d
        self.conv3X3 = ConvBN(in_channel, out_channel // 2, kernel_size=3, stride=1, padding=1, groups=1,
                              norm_layer=norm_layer)

        self.conv5X5_1 = ConvBNReLU(in_channel, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky)
        self.conv5X5_2 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.conv7X7_2 = ConvBNReLU(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                    norm_layer=norm_layer, leaky=leaky)
        self.conv7X7_3 = ConvBN(out_channel // 4, out_channel // 4, kernel_size=3, stride=1, padding=1, groups=1,
                                norm_layer=norm_layer)

        self.cat = P.Concat(axis=1)
        self.relu = nn.ReLU()

    def construct(self, x):
        conv3X3 = self.conv3X3(x)

        conv5X5_1 = self.conv5X5_1(x)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7X7_3(conv7X7_2)

        out = self.cat((conv3X3, conv5X5, conv7X7))
        out = self.relu(out)

        return out

class FPN(nn.Cell):
    def __init__(self):
        super(FPN, self).__init__()
        out_channels = 256
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        norm_layer = nn.BatchNorm2d
        self.output1 = ConvBNReLU(512, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                  norm_layer=norm_layer, leaky=leaky)
        self.output2 = ConvBNReLU(1024, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                  norm_layer=norm_layer, leaky=leaky)
        self.output3 = ConvBNReLU(2048, 256, kernel_size=1, stride=1, padding=0, groups=1,
                                  norm_layer=norm_layer, leaky=leaky)

        self.merge1 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1, groups=1,
                                 norm_layer=norm_layer, leaky=leaky)
        self.merge2 = ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1, groups=1,
                                 norm_layer=norm_layer, leaky=leaky)

    def construct(self, input1, input2, input3):
        output1 = self.output1(input1)
        output2 = self.output2(input2)
        output3 = self.output3(input3)

        up3 = P.ResizeNearestNeighbor([P.Shape()(output2)[2], P.Shape()(output2)[3]])(output3)
        output2 = up3 + output2
        output2 = self.merge2(output2)

        up2 = P.ResizeNearestNeighbor([P.Shape()(output1)[2], P.Shape()(output1)[3]])(output2)
        output1 = up2 + output1
        output1 = self.merge1(output1)

        return output1, output2, output3

class ClassHead(nn.Cell):
    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors

        weight_shape = (self.num_anchors * 2, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = Init_KaimingUniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0,
                                 has_bias=True, weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 2))

class BboxHead(nn.Cell):
    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()

        weight_shape = (num_anchors * 4, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = Init_KaimingUniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 4))

class LandmarkHead(nn.Cell):
    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()

        weight_shape = (num_anchors * 10, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = Init_KaimingUniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = P.Transpose()
        self.reshape = P.Reshape()

    def construct(self, x):
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (P.Shape()(out)[0], -1, 10))

class RetinaFace(nn.Cell):
    def __init__(self, phase='train', backbone=None):

        super(RetinaFace, self).__init__()
        self.phase = phase

        self.base = backbone

        self.fpn = FPN()

        self.ssh1 = SSH(256, 256)
        self.ssh2 = SSH(256, 256)
        self.ssh3 = SSH(256, 256)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=[256, 256, 256], anchor_num=[2, 2, 2])

        self.cat = P.Concat(axis=1)

    def _make_class_head(self, fpn_num, inchannels, anchor_num):
        classhead = nn.CellList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels[i], anchor_num[i]))
        return classhead

    def _make_bbox_head(self, fpn_num, inchannels, anchor_num):
        bboxhead = nn.CellList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels[i], anchor_num[i]))
        return bboxhead

    def _make_landmark_head(self, fpn_num, inchannels, anchor_num):
        landmarkhead = nn.CellList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels[i], anchor_num[i]))
        return landmarkhead

    def construct(self, inputs):


        f1, f2, f3 = self.base(inputs)
        f1, f2, f3 = self.fpn(f1, f2, f3)

        # SSH
        f1 = self.ssh1(f1)
        f2 = self.ssh2(f2)
        f3 = self.ssh3(f3)
        features = [f1, f2, f3]

        bbox = ()
        for i, feature in enumerate(features):
            bbox = bbox + (self.BboxHead[i](feature),)
        bbox_regressions = self.cat(bbox)

        cls = ()
        for i, feature in enumerate(features):
            cls = cls + (self.ClassHead[i](feature),)
        classifications = self.cat(cls)

        landm = ()
        for i, feature in enumerate(features):
            landm = landm + (self.LandmarkHead[i](feature),)
        ldm_regressions = self.cat(landm)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, P.Softmax(-1)(classifications), ldm_regressions)

        return output

class RetinaFaceWithLossCell(nn.Cell):
    def __init__(self, network, multibox_loss, config):
        super(RetinaFaceWithLossCell, self).__init__()
        self.network = network
        self.loc_weight = config['loc_weight']
        self.class_weight = config['class_weight']
        self.landm_weight = config['landm_weight']
        self.multibox_loss = multibox_loss

    def construct(self, img, loc_t, conf_t, landm_t):
        pred_loc, pre_conf, pre_landm = self.network(img)
        loss_loc, loss_conf, loss_landm = self.multibox_loss(pred_loc, loc_t, pre_conf, conf_t, pre_landm, landm_t)

        return loss_loc * self.loc_weight + loss_conf * self.class_weight + loss_landm * self.landm_weight

class TrainingWrapper(nn.Cell):
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = mindspore.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        class_list = [mindspore.context.ParallelMode.DATA_PARALLEL, mindspore.context.ParallelMode.HYBRID_PARALLEL]
        if self.parallel_mode in class_list:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))
