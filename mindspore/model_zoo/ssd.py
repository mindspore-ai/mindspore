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
import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.initializer import initializer
from mindspore.ops.operations import TensorAdd
from mindspore import Parameter


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same'):
    weight_shape = (out_channel, in_channel, kernel_size, kernel_size)
    weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32).to_tensor()
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=0, pad_mode=pad_mod, weight_init=weight)


def _make_divisible(v, divisor, min_value=None):
    """nsures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class DepthwiseConv(nn.Cell):
    """
    Depthwise Convolution warpper definition.

    Args:
        in_planes (int): Input channel.
        kernel_size (int): Input kernel size.
        stride (int): Stride size.
        pad_mode (str): pad mode in (pad, same, valid)
        channel_multiplier (int): Output channel multiplier
        has_bias (bool): has bias or not

    Returns:
        Tensor, output tensor.

    Examples:
        >>> DepthwiseConv(16, 3, 1, 'pad', 1, channel_multiplier=1)
    """
    def __init__(self, in_planes, kernel_size, stride, pad_mode, pad, channel_multiplier=1, has_bias=False):
        super(DepthwiseConv, self).__init__()
        self.has_bias = has_bias
        self.in_channels = in_planes
        self.channel_multiplier = channel_multiplier
        self.out_channels = in_planes * channel_multiplier
        self.kernel_size = (kernel_size, kernel_size)
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=channel_multiplier,
                                                      kernel_size=self.kernel_size,
                                                      stride=stride, pad_mode=pad_mode, pad=pad)
        self.bias_add = P.BiasAdd()
        weight_shape = [channel_multiplier, in_planes, *self.kernel_size]
        self.weight = Parameter(initializer('ones', weight_shape), name='weight')

        if has_bias:
            bias_shape = [channel_multiplier * in_planes]
            self.bias = Parameter(initializer('zeros', bias_shape), name='bias')
        else:
            self.bias = None

    def construct(self, x):
        output = self.depthwise_conv(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


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
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        if groups == 1:
            conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad',
                             padding=padding)
        else:
            conv = DepthwiseConv(in_planes, kernel_size, stride, pad_mode='pad', pad=padding)
        layers = [conv, nn.BatchNorm2d(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(nn.Cell):
    """
    Mobilenetv2 residual block definition.

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
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.SequentialCell(layers)
        self.add = TensorAdd()
        self.cast = P.Cast()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return self.add(identity, x)
        return x


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (Class): The default config of SSD.

    Returns:
        Tensor, flatten predictions.
    """
    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = config.NUM_SSD_BOXES
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

    Args:
        config (Class): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """
    def __init__(self, config):
        super(MultiBox, self).__init__()
        num_classes = config.NUM_CLASSES
        out_channels = config.EXTRAS_OUT_CHANNELS
        num_default = config.NUM_DEFAULT

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_conv2d(out_channel, 4 * num_default[k],
                                   kernel_size=3, stride=1, pad_mod='same')]
            cls_layers += [_conv2d(out_channel, num_classes * num_default[k],
                                   kernel_size=3, stride=1, pad_mod='same')]

        self.multi_loc_layers = nn.layer.CellList(loc_layers)
        self.multi_cls_layers = nn.layer.CellList(cls_layers)
        self.flatten_concat = FlattenConcat(config)

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
        config (Class): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SSD300(backbone=resnet34(num_classes=None),
                config=ConfigSSDResNet34()).
    """
    def __init__(self, backbone, config, is_training=True):
        super(SSD300, self).__init__()

        self.backbone = backbone
        in_channels = config.EXTRAS_IN_CHANNELS
        out_channels = config.EXTRAS_OUT_CHANNELS
        ratios = config.EXTRAS_RATIO
        strides = config.EXTRAS_STRIDES
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(in_channels[i], out_channels[i], stride=strides[i], expand_ratio=ratios[i])
            residual_list.append(residual)
        self.multi_residual = nn.layer.CellList(residual_list)
        self.multi_box = MultiBox(config)
        self.is_training = is_training
        if not is_training:
            self.softmax = P.Softmax()

    def construct(self, x):
        layer_out_13, output = self.backbone(x)
        multi_feature = (layer_out_13, output)
        feature = output
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.softmax(pred_label)
        return pred_loc, pred_label


class LocalizationLoss(nn.Cell):
    """"
    Computes the localization loss with SmoothL1Loss.

    Returns:
        Tensor, box regression loss.
    """
    def __init__(self):
        super(LocalizationLoss, self).__init__()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.loss = nn.SmoothL1Loss()
        self.expand_dims = P.ExpandDims()
        self.less = P.Less()

    def construct(self, pred_loc, gt_loc, gt_label, num_matched_boxes):
        mask = F.cast(self.less(0, gt_label), mstype.float32)
        mask = self.expand_dims(mask, -1)
        smooth_l1 = self.loss(gt_loc, pred_loc) * mask
        box_loss = self.reduce_sum(smooth_l1, 1)
        return self.reduce_mean(box_loss / F.cast(num_matched_boxes, mstype.float32), (0, 1))


class ClassificationLoss(nn.Cell):
    """"
    Computes the classification loss with hard example mining.

    Args:
        config (Class): The default config of SSD.

    Returns:
        Tensor, classification loss.
    """
    def __init__(self, config):
        super(ClassificationLoss, self).__init__()
        self.num_classes = config.NUM_CLASSES
        self.num_boxes = config.NUM_SSD_BOXES
        self.neg_pre_positive = config.NEG_PRE_POSITIVE
        self.minimum = P.Minimum()
        self.less = P.Less()
        self.sort = P.TopK()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.sort_descend = P.TopK(True)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, pred_label, gt_label, num_matched_boxes):
        gt_label = F.cast(gt_label, mstype.int32)
        mask = F.cast(self.less(0, gt_label), mstype.float32)
        gt_label_shape = F.shape(gt_label)
        pred_label = F.reshape(pred_label, (-1, self.num_classes))
        gt_label = F.reshape(gt_label, (-1,))
        cross_entropy = self.cross_entropy(pred_label, gt_label)
        cross_entropy = F.reshape(cross_entropy, gt_label_shape)

        # Hard example mining
        num_matched_boxes = F.reshape(num_matched_boxes, (-1,))
        neg_masked_cross_entropy = F.cast(cross_entropy * (1- mask), mstype.float16)
        _, loss_idx = self.sort_descend(neg_masked_cross_entropy, self.num_boxes)
        _, relative_position = self.sort(F.cast(loss_idx, mstype.float16), self.num_boxes)
        num_neg_boxes = self.minimum(num_matched_boxes * self.neg_pre_positive, self.num_boxes)
        tile_num_neg_boxes = self.tile(self.expand_dims(num_neg_boxes, -1), (1, self.num_boxes))
        top_k_neg_mask = F.cast(self.less(relative_position, tile_num_neg_boxes), mstype.float32)
        class_loss = self.reduce_sum(cross_entropy * (mask + top_k_neg_mask), 1)
        return self.reduce_mean(class_loss / F.cast(num_matched_boxes, mstype.float32), 0)


class SSDWithLossCell(nn.Cell):
    """"
    Provide SSD training loss through network.

    Args:
        network (Cell): The training network.
        config (Class): SSD config.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, network, config):
        super(SSDWithLossCell, self).__init__()
        self.network = network
        self.class_loss = ClassificationLoss(config)
        self.box_loss = LocalizationLoss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        pred_loc, pred_label = self.network(x)
        loss_cls = self.class_loss(pred_label, gt_label, num_matched_boxes)
        loss_loc = self.box_loss(pred_loc, gt_loc, gt_label, num_matched_boxes)
        return loss_cls + loss_loc


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
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ms.ParallelMode.DATA_PARALLEL, ms.ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("mirror_mean")
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



class SSDWithMobileNetV2(nn.Cell):
    """
    MobileNetV2 architecture for SSD backbone.

    Args:
        width_mult (int): Channels multiplier for round to 8/16 and others. Default is 1.
        inverted_residual_setting (list): Inverted residual settings. Default is None
        round_nearest (list): Channel round to. Default is 8
    Returns:
        Tensor, the 13th feature after ConvBNReLU in MobileNetV2.
        Tensor, the last feature in MobileNetV2.

    Examples:
        >>> SSDWithMobileNetV2()
    """
    def __init__(self, width_mult=1.0, inverted_residual_setting=None, round_nearest=8):
        super(SSDWithMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        if len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        #building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        layer_index = 0
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if layer_index == 13:
                    hidden_dim = int(round(input_channel * t))
                    self.expand_layer_conv_13 = ConvBNReLU(input_channel, hidden_dim, kernel_size=1)
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                layer_index += 1
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features_1 = nn.SequentialCell(features[:14])
        self.features_2 = nn.SequentialCell(features[14:])

    def construct(self, x):
        out = self.features_1(x)
        expand_layer_conv_13 = self.expand_layer_conv_13(out)
        out = self.features_2(out)
        return expand_layer_conv_13, out

    def get_out_channels(self):
        return self.last_channel

def ssd_mobilenet_v2(**kwargs):
    return SSDWithMobileNetV2(**kwargs)
