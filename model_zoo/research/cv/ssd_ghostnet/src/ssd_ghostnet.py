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

"""SSD net based MobilenetV2."""
import math
import numpy as np

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from src.model_utils.config import config


def _make_divisible(x, divisor=4):
    return int(np.ceil(x * 1. / divisor) * divisor)


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same'):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=0, pad_mode=pad_mod, has_bias=True)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    depthwise_conv = nn.Conv2d(in_channel, in_channel, kernel_size, stride, pad_mode='same', padding=pad,
                               has_bias=False, group=in_channel, weight_init='ones')
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return nn.SequentialCell([depthwise_conv, _bn(in_channel), nn.ReLU6(), conv])


class ConvBNReLU(nn.Cell):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthiwse is input channel. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, use_act=True, act_type='relu'):
        super(ConvBNReLU, self).__init__()
        padding = 0
        if groups == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='same',
                             padding=padding)
        else:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='same', padding=padding,
                             has_bias=False, group=groups, weight_init='ones')
        layers = [conv, _bn(out_planes)]
        if use_act:
            layers.append(Activation(act_type))
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class MyHSigmoid(nn.Cell):
    def __init__(self):
        super(MyHSigmoid, self).__init__()
        self.relu6 = nn.ReLU6()

    def construct(self, x):
        return self.relu6(x + 3.) / 6.


class Activation(nn.Cell):
    """
    Activation definition.

    Args:
        act_func(string): activation name.

    Returns:
         Tensor, output tensor.
    """

    def __init__(self, act_func):
        super(Activation, self).__init__()
        if act_func == 'relu':
            self.act = nn.ReLU()
        elif act_func == 'relu6':
            self.act = nn.ReLU6()
        elif act_func in ('hsigmoid', 'hard_sigmoid'):
            self.act = MyHSigmoid()  # nn.HSigmoid()
        elif act_func in ('hswish', 'hard_swish'):
            self.act = nn.HSwish()
        else:
            raise NotImplementedError

    def construct(self, x):
        return self.act(x)


class GlobalAvgPooling(nn.Cell):
    """
    Global avg pooling definition.

    Args:

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GlobalAvgPooling()
    """

    def __init__(self, keep_dims=False):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=keep_dims)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class SE(nn.Cell):
    """
    SE warpper definition.

    Args:
        num_out (int): Output channel.
        ratio (int): middle output ratio.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> SE(4)
    """

    def __init__(self, num_out, ratio=4):
        super(SE, self).__init__()
        num_mid = _make_divisible(num_out // ratio)
        self.pool = GlobalAvgPooling(keep_dims=True)
        self.conv_reduce = nn.Conv2d(in_channels=num_out, out_channels=num_mid,
                                     kernel_size=1, has_bias=True, pad_mode='pad')
        self.act1 = Activation('relu')
        self.conv_expand = nn.Conv2d(in_channels=num_mid, out_channels=num_out,
                                     kernel_size=1, has_bias=True, pad_mode='pad')
        self.act2 = Activation('hsigmoid')
        self.mul = P.Mul()

    def construct(self, x):
        out = self.pool(x)
        out = self.conv_reduce(out)
        out = self.act1(out)
        out = self.conv_expand(out)
        out = self.act2(out)
        out = self.mul(x, out)
        return out


class GhostModule(nn.Cell):
    """
    GhostModule warpper definition.

    Args:
        num_in (int): Input channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        padding (int): Padding number.
        use_act (bool): Used activation or not.
        act_type (string): Activation type.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostModule(3, 3)
    """

    def __init__(self, num_in, num_out, kernel_size=1, stride=1, padding=0, ratio=2, dw_size=3,
                 use_act=True, act_type='relu'):
        super(GhostModule, self).__init__()
        init_channels = math.ceil(num_out / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvBNReLU(num_in, init_channels, kernel_size=kernel_size, stride=stride,
                                       groups=1, use_act=use_act, act_type='relu')
        self.cheap_operation = ConvBNReLU(init_channels, new_channels, kernel_size=dw_size, stride=1,
                                          groups=init_channels, use_act=use_act, act_type='relu')
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return self.concat((x1, x2))


class GhostBottleneck(nn.Cell):
    """
    GhostBottleneck warpper definition.

    Args:
        num_in (int): Input channel.
        num_mid (int): Middle channel.
        num_out (int): Output channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        act_type (str): Activation type.
        use_se (bool): Use SE warpper or not.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> GhostBottleneck(16, 3, 1, 1)
    """

    def __init__(self, num_in, num_mid, num_out, kernel_size, stride=1, act_type='relu', use_se=False,
                 use_res_connect=True, last_relu=False):
        super(GhostBottleneck, self).__init__()
        self.ghost1 = GhostModule(num_in, num_mid, kernel_size=1,
                                  stride=1, padding=0, act_type=act_type)

        self.use_res_connect = use_res_connect
        self.last_relu = last_relu
        self.use_dw = stride > 1
        self.dw = None
        if self.use_dw:
            self.dw = ConvBNReLU(num_mid, num_mid, kernel_size=kernel_size, stride=stride,
                                 act_type=act_type, groups=num_mid, use_act=False)

        self.use_se = use_se
        if use_se:
            self.se = SE(num_mid)

        self.ghost2 = GhostModule(num_mid, num_out, kernel_size=1, stride=1,
                                  padding=0, act_type=act_type, use_act=False)
        self.relu = nn.ReLU()
        if self.use_res_connect:
            self.down_sample = False
            if num_in != num_out or stride != 1:
                self.down_sample = True
            self.shortcut = None
            if self.down_sample:
                self.shortcut = nn.SequentialCell([
                    ConvBNReLU(num_in, num_in, kernel_size=kernel_size, stride=stride,
                               groups=num_in, use_act=False),
                    ConvBNReLU(num_in, num_out, kernel_size=1, stride=1,
                               groups=1, use_act=False),
                ])
            self.add = P.Add()

    def construct(self, x):
        """construct"""
        shortcut = x
        out = self.ghost1(x)
        if self.use_dw:
            out = self.dw(out)
        if self.use_se:
            out = self.se(out)
        out = self.ghost2(out)
        if self.use_res_connect:
            if self.down_sample:
                shortcut = self.shortcut(shortcut)
            out = self.add(shortcut, out)
        if self.last_relu:
            out = self.relu(out)
        return out

    def _get_pad(self, kernel_size):
        """set the padding number"""
        pad = 0
        if kernel_size == 1:
            pad = 0
        elif kernel_size == 3:
            pad = 1
        elif kernel_size == 5:
            pad = 2
        elif kernel_size == 7:
            pad = 3
        else:
            raise NotImplementedError
        return pad


class InvertedResidual(nn.Cell):
    """
    Residual block definition.

    Args:
        inp (int): Input channel.
        oup (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        expand_ratio (int): expand ration of input channel

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, 1, 1)
    """

    def __init__(self, inp, oup, stride, expand_ratio, last_relu=False):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim,
                       stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1,
                      stride=1, has_bias=False),
            _bn(oup),
        ])
        self.conv = nn.SequentialCell(layers)
        self.add = P.Add()
        self.cast = P.Cast()
        self.last_relu = last_relu
        self.relu = nn.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            x = self.add(identity, x)
        if self.last_relu:
            x = self.relu(x)
        return x


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = config.num_ssd_boxes
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = F.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (F.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return F.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self):
        super(MultiBox, self).__init__()
        num_classes = config.num_classes
        out_channels = config.extras_out_channels
        num_default = config.num_default

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_last_conv2d(out_channel, 4 * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]
            cls_layers += [_last_conv2d(out_channel, num_classes * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]

        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.flatten_concat = FlattenConcat()

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class SSD300(nn.Cell):
    """
    SSD300 Network. Default backbone is resnet34.

    Args:
        backbone (Cell): Backbone Network.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SSD300(backbone=resnet34(num_classes=None)).
    """

    def __init__(self, backbone, is_training=True, **kwargs):
        super(SSD300, self).__init__()

        self.backbone = backbone
        in_channels = config.extras_in_channels
        out_channels = config.extras_out_channels
        ratios = config.extras_ratio
        strides = config.extras_srides
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(in_channels[i], out_channels[i], stride=strides[i],
                                        expand_ratio=ratios[i], last_relu=True)
            residual_list.append(residual)
        self.multi_residual = nn.layer.CellList(residual_list)
        self.multi_box = MultiBox()
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        r"""construct of SSD300"""
        layer_out_11, output = self.backbone(x)
        multi_feature = (layer_out_11, output)
        feature = output
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        return pred_loc, pred_label


class SigmoidFocalClassificationLoss(nn.Cell):
    """"
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        r"""construct of SigmoidFocalClassificationLoss"""
        label = self.onehot(label, F.shape(
            logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = F.cast(label, mstype.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + \
            (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSDWithLossCell(nn.Cell):
    """"
    Provide SSD training loss through network.

    Args:
        network (Cell): The training network.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, network):
        super(SSDWithLossCell, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(
            config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        r"""construct of SSDWithLossCell"""
        pred_loc, pred_label = self.network(x)
        mask = F.cast(self.less(0, gt_label), mstype.float32)
        num_matched_boxes = self.reduce_sum(
            F.cast(num_matched_boxes, mstype.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_mean(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [context.ParallelMode.DATA_PARALLEL, context.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(
                optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


class SSDWithGhostNet(nn.Cell):
    """
    Ghostnett architecture for SSD backbone.

    Args:
        model_cfgs (list): model config
        multiplier (int): Channels multiplier for round to 8/16 and others. Default is 1.
        round_nearest (list): Channel round to. Default is 8
    Returns:
        Tensor, the 11th feature after ConvBNReLU in MobileNetV2.
        Tensor, the last feature in MobileNetV2.

    Examples:
        >>> SSDWithGhostNet()
    """

    def __init__(self, model_cfgs, multiplier=1., round_nearest=8):
        super(SSDWithGhostNet, self).__init__()
        self.cfgs = model_cfgs['cfg']
        self.inplanes = 20  # for "1.3x"
        first_conv_in_channel = 3
        first_conv_out_channel = _make_divisible(multiplier * self.inplanes)

        blocks = []
        blocks.append(ConvBNReLU(first_conv_in_channel, first_conv_out_channel,
                                 kernel_size=3, stride=2, groups=1, use_act=True, act_type='relu'))

        layer_index = 0
        for layer_cfg in self.cfgs:
            if layer_index == 11:
                hidden_dim = int(round(self.inplanes * 6))
                self.expand_layer_conv_11 = ConvBNReLU(
                    self.inplanes, hidden_dim, kernel_size=1)
            blocks.append(self._make_layer(kernel_size=layer_cfg[0],
                                           exp_ch=_make_divisible(
                                               multiplier * layer_cfg[1]),
                                           out_channel=_make_divisible(
                                               multiplier * layer_cfg[2]),
                                           use_se=layer_cfg[3],
                                           act_func=layer_cfg[4],
                                           stride=layer_cfg[5]))
            layer_index += 1
        output_channel = _make_divisible(
            multiplier * model_cfgs["cls_ch_squeeze"])
        blocks.append(ConvBNReLU(_make_divisible(multiplier * self.cfgs[-1][2]), output_channel,
                                 kernel_size=1, stride=1, groups=1, use_act=True))

        self.features_1 = nn.SequentialCell(blocks[:12])
        self.features_2 = nn.SequentialCell(blocks[12:])

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, stride=1):
        mid_planes = exp_ch
        out_planes = out_channel
        layer = GhostBottleneck(self.inplanes, mid_planes, out_planes,
                                kernel_size, stride=stride, act_type=act_func, use_se=use_se)
        self.inplanes = out_planes
        return layer

    def construct(self, x):
        out = self.features_1(x)
        expand_layer_conv_11 = self.expand_layer_conv_11(out)
        out = self.features_2(out)
        return expand_layer_conv_11, out


def ssd_ghostnet(**kwargs):
    """
    Constructs a SSD GhostNet model
    """
    model_cfgs = {
        "1x": {
            "cfg": [
                # k, exp, c,  se,    nl,   s,
                # stage1
                [3, 16, 16, False, 'relu', 1],
                # stage2
                [3, 48, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                # stage3
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                # stage4
                [3, 240, 80, False, 'relu', 2],
                [3, 200, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 184, 80, False, 'relu', 1],
                [3, 480, 112, True, 'relu', 1],
                [3, 672, 112, True, 'relu', 1],
                # stage5
                [5, 672, 160, True, 'relu', 2],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, True, 'relu', 1],
                [5, 960, 160, False, 'relu', 1],
                [5, 960, 160, True, 'relu', 1]],
            "cls_ch_squeeze": 960,
        },
        "1.3x": {
            "cfg": [
                # k, exp, c,  se,    nl,   s
                # stage1
                [3, 24, 20, False, 'relu', 1],
                # stage2
                [3, 64, 32, False, 'relu', 2],
                [3, 96, 32, False, 'relu', 1],
                # stage3
                [5, 96, 52, True, 'relu', 2],
                [5, 160, 52, True, 'relu', 1],
                # stage4
                [3, 312, 104, False, 'relu', 2],
                [3, 264, 104, False, 'relu', 1],
                [3, 240, 104, False, 'relu', 1],
                [3, 240, 104, False, 'relu', 1],
                [3, 624, 144, True, 'relu', 1],
                [3, 864, 144, True, 'relu', 1],
                # stage5
                [5, 864, 208, True, 'relu', 2],
                [5, 1248, 208, False, 'relu', 1],
                [5, 1248, 208, True, 'relu', 1],
                [5, 1248, 208, False, 'relu', 1],
                [5, 1248, 208, True, 'relu', 1]],
            "cls_ch_squeeze": 1248,
        }
    }
    return SSDWithGhostNet(model_cfgs["1.3x"], **kwargs)
