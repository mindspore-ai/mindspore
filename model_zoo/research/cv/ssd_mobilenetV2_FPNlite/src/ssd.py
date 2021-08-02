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
from src.mobilenet_v2_fpn import ConvBNReLU, _conv2d
from .mobilenet_v2_fpn import mobilenet_v2_fpn

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
                _conv2d(out_channels, out_channels, 3, 1, out_channels) for x in range(num_addition_layers)
            ]
            cls_convs = [
                _conv2d(out_channels, out_channels, 3, 1, out_channels) for x in range(num_addition_layers)
            ]
            addition_loc_layer_list = []
            addition_cls_layer_list = []
            for _ in range(num_features):
                addition_loc_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, out_channels,
                               loc_convs[x]) for x in range(num_addition_layers)
                ]
                addition_cls_layer = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, out_channels,
                               cls_convs[x]) for x in range(num_addition_layers)
                ]
                addition_loc_layer_list.append(nn.SequentialCell(addition_loc_layer))
                addition_cls_layer_list.append(nn.SequentialCell(addition_cls_layer))
            self.addition_layer_loc = nn.CellList(addition_loc_layer_list)
            self.addition_layer_cls = nn.CellList(addition_cls_layer_list)
        else:
            convs = [
                _conv2d(out_channels, out_channels, 3, 1, out_channels) for x in range(num_addition_layers)
            ]
            addition_layer_list = []
            for _ in range(num_features):
                addition_layers = [
                    ConvBNReLU(out_channels, out_channels, 3, 1, 1, convs[x]) for x in range(num_addition_layers)
                ]
                addition_layer_list.append(nn.SequentialCell(addition_layers))
            self.addition_layer = nn.SequentialCell(addition_layer_list)

        loc_layers = [_conv2d(out_channels, 4 * num_default,
                              kernel_size=3, stride=1, group=out_channels, pad_mod='same')]
        cls_layers = [_conv2d(out_channels, num_classes * num_default,
                              kernel_size=3, stride=1, group=out_channels, pad_mod='same')]

        self.loc_layers = nn.SequentialCell(loc_layers)
        self.cls_layers = nn.SequentialCell(cls_layers)
        self.flatten_concat = FlattenConcat(config)

    def construct(self, inputs):
        """get localization predictions and class conf scores"""
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



class SsdMobilenetV2Fpn(nn.Cell):
    """
    SSD Network using mobilenetV1 with fpn to extract features

    Args:
        config (dict): The default config of SSD.
        is_training (bool): Used for training, default is True.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.

    Examples:backbone
         SsdMobilenetV2Fpn(config, True).
    """
    def __init__(self, config):
        super(SsdMobilenetV2Fpn, self).__init__()
        self.multi_box = WeightSharedMultiBox(config)
        self.activation = P.Sigmoid()
        self.feature_extractor = mobilenet_v2_fpn(config)

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
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        """get loss"""
        pred_loc, pred_label = self.network(x)
        mask = F.cast(self.less(0, gt_label), mstype.float32)
        num_matched_boxes = self.reduce_sum(F.cast(num_matched_boxes, mstype.float32))

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_mean(smooth_l1, -1), -1)

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
        """opt"""
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
        self.optimizer(grads)
        return loss



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
        """get pred_xy and pred_label"""
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


def ssd_mobilenet_v2_fpn(**kwargs):
    return SsdMobilenetV2Fpn(**kwargs)
