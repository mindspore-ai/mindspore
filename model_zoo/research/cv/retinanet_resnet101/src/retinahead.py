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

"""retinanet based resnet."""

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

from .bottleneck import FPN


class FlattenConcat(nn.Cell):
    """
    Concatenate predictions into a single tensor.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, config):
        super(FlattenConcat, self).__init__()
        self.num_retinanet_boxes = config.num_retinanet_boxes
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = F.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (F.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return F.reshape(res, (batch_size, self.num_retinanet_boxes, -1))


def ClassificationModel(in_channel, num_anchors, kernel_size=3, stride=1, pad_mod='same', num_classes=81,
                        feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv5 = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])


def RegressionModel(in_channel, num_anchors, kernel_size=3, stride=1, pad_mod='same', feature_size=256):
    conv1 = nn.Conv2d(in_channel, feature_size, kernel_size=3, pad_mode='same')
    conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, pad_mode='same')
    conv5 = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, pad_mode='same')
    return nn.SequentialCell([conv1, nn.ReLU(), conv2, nn.ReLU(), conv3, nn.ReLU(), conv4, nn.ReLU(), conv5])


class MultiBox(nn.Cell):
    """
    Multibox conv layers. Each multibox layer contains class conf scores and localization predictions.

    Args:
        config (dict): The default config of retinanet.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, config):
        super(MultiBox, self).__init__()

        out_channels = config.extras_out_channels
        num_default = config.num_default
        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [RegressionModel(in_channel=out_channel, num_anchors=num_default[k])]
            cls_layers += [ClassificationModel(in_channel=out_channel, num_anchors=num_default[k])]

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


class retinahead(nn.Cell):
    """retinahead"""
    def __init__(self, backbone, config, is_training=True):
        super(retinahead, self).__init__()

        self.fpn = FPN(backbone=backbone, config=config)
        self.multi_box = MultiBox(config)
        self.is_training = is_training
        if not is_training:
            self.activation = P.Sigmoid()

    def construct(self, inputs):
        features = self.fpn(inputs)
        pred_loc, pred_label = self.multi_box(features)
        return pred_loc, pred_label


class retinanetWithLossCell(nn.Cell):
    """"
    Provide retinanet training loss through network.

    Args:
        network (Cell): The training network.
        config (dict): retinanet config.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, network, config):
        super(retinanetWithLossCell, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(config.gamma, config.alpha)
        self.loc_loss = nn.SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        """construct"""
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


class TrainingWrapper(nn.Cell):
    """
    Encapsulation class of retinanet network training.

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
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
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


class retinanetInferWithDecoder(nn.Cell):
    """
    retinanet Infer wrapper to decode the bbox locations.

    Args:
        network (Cell): the origin retinanet infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator
        config (dict): retinanet config
    Returns:
        Tensor, the locations for bbox after decoder representing (y0,x0,y1,x1)
        Tensor, the prediction labels.

    """
    def __init__(self, network, default_boxes, config):
        super(retinanetInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = default_boxes
        self.prior_scaling_xy = config.prior_scaling[0]
        self.prior_scaling_wh = config.prior_scaling[1]

    def construct(self, x):
        """construct"""
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
