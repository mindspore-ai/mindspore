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
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from .fpn import mobilenet_v1_fpn, resnet50_fpn
from .vgg16 import vgg16


def _make_divisible(v, divisor, min_value=None):
    """nsures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same'):
    return nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                     padding=0, pad_mode=pad_mod, has_bias=True)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                               padding=pad, group=in_channels)
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
        shared_conv(Cell): Use the weight shared conv, default: None.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, shared_conv=None):
        super(ConvBNReLU, self).__init__()
        padding = 0
        in_channels = in_planes
        out_channels = out_planes
        if shared_conv is None:
            if groups == 1:
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same', padding=padding)
            else:
                out_channels = in_planes
                conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode='same',
                                 padding=padding, group=in_channels)
            layers = [conv, _bn(out_planes), nn.ReLU6()]
        else:
            layers = [shared_conv, _bn(out_planes), nn.ReLU6()]
        self.features = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.features(x)
        return output


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
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
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

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, flatten predictions.
    """
    def __init__(self, config):
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

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """
    def __init__(self, config):
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
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class WeightSharedMultiBox(nn.Cell):
    """
    Weight shared Multi-box conv layers. Each multi-box layer contains class conf scores and localization predictions.
    All box predictors shares the same conv weight in different features.

    Args:
        config (dict): The default config of SSD.
        loc_cls_shared_addition(bool): Whether the location predictor and classifier prediction share the
                                       same addition layer.
    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """
    def __init__(self, config, loc_cls_shared_addition=False):
        super(WeightSharedMultiBox, self).__init__()
        num_classes = config.num_classes
        out_channels = config.extras_out_channels[0]
        num_default = config.num_default[0]
        num_features = len(config.feature_size)
        num_addition_layers = config.num_addition_layers
        self.loc_cls_shared_addition = loc_cls_shared_addition

        if not loc_cls_shared_addition:
            loc_convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            cls_convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            addition_loc_layer_list = []
            addition_cls_layer_list = []
            for _ in range(num_features):
                addition_loc_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, loc_convs[x]) for x in range(num_addition_layers)
                ]
                addition_cls_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, cls_convs[x]) for x in range(num_addition_layers)
                ]
                addition_loc_layer_list.append(nn.SequentialCell(addition_loc_layer))
                addition_cls_layer_list.append(nn.SequentialCell(addition_cls_layer))
            self.addition_layer_loc = nn.CellList(addition_loc_layer_list)
            self.addition_layer_cls = nn.CellList(addition_cls_layer_list)
        else:
            convs = [
                _conv2d(out_channels, out_channels, 3, 1) for x in range(num_addition_layers)
            ]
            addition_layer_list = []
            for _ in range(num_features):
                addition_layers = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, convs[x]) for x in range(num_addition_layers)
                ]
                addition_layer_list.append(nn.SequentialCell(addition_layers))
            self.addition_layer = nn.SequentialCell(addition_layer_list)

        loc_layers = [_conv2d(out_channels, 4 * num_default,
                              kernel_size=3, stride=1, pad_mod='same')]
        cls_layers = [_conv2d(out_channels, num_classes * num_default,
                              kernel_size=3, stride=1, pad_mod='same')]

        self.loc_layers = nn.SequentialCell(loc_layers)
        self.cls_layers = nn.SequentialCell(cls_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        num_heads = len(inputs)
        for i in range(num_heads):
            if self.loc_cls_shared_addition:
                features = self.addition_layer[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                cls_outputs += (self.cls_layers(features),)
            else:
                features = self.addition_layer_loc[i](inputs[i])
                loc_outputs += (self.loc_layers(features),)
                features = self.addition_layer_cls[i](inputs[i])
                cls_outputs += (self.cls_layers(features),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class SSD300(nn.Cell):
    """
    SSD300 Network. Default backbone is resnet34.

    Args:
        backbone (Cell): Backbone Network.
        config (dict): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SSD300(backbone=resnet34(num_classes=None),
                config=config).
    """
    def __init__(self, backbone, config, is_training=True):
        super(SSD300, self).__init__()

        self.backbone = backbone
        in_channels = config.extras_in_channels
        out_channels = config.extras_out_channels
        ratios = config.extras_ratio
        strides = config.extras_strides
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(in_channels[i], out_channels[i], stride=strides[i],
                                        expand_ratio=ratios[i], last_relu=True)
            residual_list.append(residual)
        self.multi_residual = nn.layer.CellList(residual_list)
        self.multi_box = MultiBox(config)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        layer_out_13, output = self.backbone(x)
        multi_feature = (layer_out_13, output)
        feature = output
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, mstype.float32)
        pred_label = F.cast(pred_label, mstype.float32)
        return pred_loc, pred_label


class SsdMobilenetV1Fpn(nn.Cell):
    """
    SSD Network using mobilenetV1 with fpn to extract features

    Args:
        config (dict): The default config of SSD.
        is_training (bool): Used for training, default is True.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SsdMobilenetV1Fpn(config, True).
    """
    def __init__(self, config):
        super(SsdMobilenetV1Fpn, self).__init__()
        self.multi_box = WeightSharedMultiBox(config)
        self.activation = P.Sigmoid()
        self.feature_extractor = mobilenet_v1_fpn(config)

    def construct(self, x):
        features = self.feature_extractor(x)
        pred_loc, pred_label = self.multi_box(features)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, mstype.float32)
        pred_label = F.cast(pred_label, mstype.float32)
        return pred_loc, pred_label

class SsdResNet50Fpn(nn.Cell):
    """
    SSD Network using ResNet50 with fpn to extract features

    Args:
        config (dict): The default config of SSD.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SsdResNet50Fpn(config).
    """
    def __init__(self, config):
        super(SsdResNet50Fpn, self).__init__()
        self.multi_box = WeightSharedMultiBox(config)
        self.activation = P.Sigmoid()
        self.feature_extractor = resnet50_fpn()

    def construct(self, x):
        features = self.feature_extractor(x)
        pred_loc, pred_label = self.multi_box(features)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, mstype.float32)
        pred_label = F.cast(pred_label, mstype.float32)
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
        label = self.onehot(label, F.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = F.cast(label, mstype.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSDWithLossCell(nn.Cell):
    """"
    Provide SSD training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): SSD config.

    Returns:
        Tensor, the loss of the network.
    """
    def __init__(self, network, config):
        super(SSDWithLossCell, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        pred_loc, pred_label = self.network(x)
        mask = F.cast(self.less(0, gt_label), mstype.float32)
        num_matched_boxes = self.reduce_sum(F.cast(num_matched_boxes, mstype.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_sum(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)


grad_scale = C.MultitypeFuncGraph("grad_scale")
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * P.Reciprocal()(scale)


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
        use_global_nrom(bool): Whether apply global norm before optimizer. Default: False
    """
    def __init__(self, network, optimizer, sens=1.0, use_global_norm=False):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.use_global_norm = use_global_norm
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)
        self.hyper_map = C.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        if self.use_global_norm:
            grads = self.hyper_map(F.partial(grad_scale, F.scalar_to_array(self.sens)), grads)
            grads = C.clip_by_global_norm(grads)
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


class SsdInferWithDecoder(nn.Cell):
    """
    SSD Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin ssd infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): ssd config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """
    def __init__(self, network, default_boxes, config):
        super(SsdInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = config.prior_scaling[0]
        self.prior_scaling_wh = config.prior_scaling[1]

    def construct(self, x):
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = P.Exp()(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = P.Concat(-1)((pred_xy_0, pred_xy_1))
        pred_xy = P.Maximum()(pred_xy, 0)
        pred_xy = P.Minimum()(pred_xy, 1)
        return pred_xy, pred_label


def ssd_mobilenet_v1_fpn(**kwargs):
    return SsdMobilenetV1Fpn(**kwargs)

def ssd_resnet50_fpn(**kwargs):
    return SsdResNet50Fpn(**kwargs)

def ssd_mobilenet_v2(**kwargs):
    return SSDWithMobileNetV2(**kwargs)


class SSD300VGG16(nn.Cell):
    def __init__(self, config):
        super(SSD300VGG16, self).__init__()

        # VGG16 backbone: block1~5
        self.backbone = vgg16()

        # SSD blocks: block6~7
        self.b6_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6, pad_mode='pad')
        self.b6_2 = nn.Dropout(0.5)

        self.b7_1 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
        self.b7_2 = nn.Dropout(0.5)

        # Extra Feature Layers: block8~11
        self.b8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, padding=1, pad_mode='pad')
        self.b8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, pad_mode='valid')

        self.b9_1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, padding=1, pad_mode='pad')
        self.b9_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, pad_mode='valid')

        self.b10_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b10_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        self.b11_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)
        self.b11_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, pad_mode='valid')

        # boxes
        self.multi_box = MultiBox(config)
        if not self.training:
            self.activation = P.Sigmoid()

    def construct(self, x):
        # VGG16 backbone: block1~5
        block4, x = self.backbone(x)

        # SSD blocks: block6~7
        x = self.b6_1(x)  # 1024
        x = self.b6_2(x)

        x = self.b7_1(x)  # 1024
        x = self.b7_2(x)
        block7 = x

        # Extra Feature Layers: block8~11
        x = self.b8_1(x)  # 256
        x = self.b8_2(x)  # 512
        block8 = x

        x = self.b9_1(x)  # 128
        x = self.b9_2(x)  # 256
        block9 = x

        x = self.b10_1(x)  # 128
        x = self.b10_2(x)  # 256
        block10 = x

        x = self.b11_1(x)  # 128
        x = self.b11_2(x)  # 256
        block11 = x

        # boxes
        multi_feature = (block4, block7, block8, block9, block10, block11)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.training:
            pred_label = self.activation(pred_label)
        pred_loc = F.cast(pred_loc, mstype.float32)
        pred_label = F.cast(pred_label, mstype.float32)
        return pred_loc, pred_label


def ssd_vgg16(**kwargs):
    return SSD300VGG16(**kwargs)
