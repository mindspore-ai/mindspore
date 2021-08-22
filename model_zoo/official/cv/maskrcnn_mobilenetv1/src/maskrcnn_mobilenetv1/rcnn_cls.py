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
"""MaskRcnn Rcnn classification and box regression network."""

import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter

class DenseNoTranpose(nn.Cell):
    """Dense method"""
    def __init__(self, input_channels, output_channels, weight_init):
        super(DenseNoTranpose, self).__init__()
        self.weight = Parameter(initializer(weight_init, [input_channels, output_channels], mstype.float32),
                                name="weight")
        self.bias = Parameter(initializer("zeros", [output_channels], mstype.float32), name="bias")
        self.matmul = P.MatMul(transpose_b=False)
        self.bias_add = P.BiasAdd()

    def construct(self, x):
        output = self.bias_add(self.matmul(x, self.weight), self.bias)
        return output

class FpnCls(nn.Cell):
    """dense layer of classification and box head"""
    def __init__(self, input_channels, output_channels, num_classes, pool_size):
        super(FpnCls, self).__init__()
        representation_size = input_channels * pool_size * pool_size
        shape_0 = (output_channels, representation_size)
        weights_0 = initializer("XavierUniform", shape=shape_0[::-1], dtype=mstype.float32)
        shape_1 = (output_channels, output_channels)
        weights_1 = initializer("XavierUniform", shape=shape_1[::-1], dtype=mstype.float32)
        self.shared_fc_0 = DenseNoTranpose(representation_size, output_channels, weights_0).to_float(mstype.float16)
        self.shared_fc_1 = DenseNoTranpose(output_channels, output_channels, weights_1).to_float(mstype.float16)

        cls_weight = initializer('Normal', shape=[num_classes, output_channels][::-1],
                                 dtype=mstype.float32)
        reg_weight = initializer('Normal', shape=[num_classes * 4, output_channels][::-1],
                                 dtype=mstype.float32)
        self.cls_scores = DenseNoTranpose(output_channels, num_classes, cls_weight).to_float(mstype.float16)
        self.reg_scores = DenseNoTranpose(output_channels, num_classes * 4, reg_weight).to_float(mstype.float16)

        self.relu = P.ReLU()
        self.flatten = P.Flatten()

    def construct(self, x):
        # two share fc layer
        x = self.flatten(x)

        x = self.relu(self.shared_fc_0(x))
        x = self.relu(self.shared_fc_1(x))

        # classifier head
        cls_scores = self.cls_scores(x)
        # bbox head
        reg_scores = self.reg_scores(x)

        return cls_scores, reg_scores

class RcnnCls(nn.Cell):
    """
    Rcnn for classification and box regression subnet.

    Args:
        config (dict) - Config.
        batch_size (int) - Batchsize.
        num_classes (int) - Class number.
        target_means (list) - Means for encode function. Default: (.0, .0, .0, .0]).
        target_stds (list) - Stds for encode function. Default: (0.1, 0.1, 0.2, 0.2).

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        RcnnCls(config=config, representation_size = 1024, batch_size=2, num_classes = 81, \
             target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    """
    def __init__(self,
                 config,
                 batch_size,
                 num_classes,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(RcnnCls, self).__init__()
        cfg = config
        self.rcnn_loss_cls_weight = Tensor(np.array(cfg.rcnn_loss_cls_weight).astype(np.float16))
        self.rcnn_loss_reg_weight = Tensor(np.array(cfg.rcnn_loss_reg_weight).astype(np.float16))
        self.rcnn_fc_out_channels = cfg.rcnn_fc_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_classes = num_classes
        self.in_channels = cfg.rcnn_in_channels
        self.train_batch_size = batch_size
        self.test_batch_size = cfg.test_batch_size

        self.fpn_cls = FpnCls(self.in_channels, self.rcnn_fc_out_channels, self.num_classes, cfg.roi_layer["out_size"])
        self.relu = P.ReLU()
        self.logicaland = P.LogicalAnd()
        self.loss_cls = P.SoftmaxCrossEntropyWithLogits()
        self.loss_bbox = P.SmoothL1Loss(beta=1.0)
        self.loss_mask = P.SigmoidCrossEntropyWithLogits()
        self.reshape = P.Reshape()
        self.onehot = P.OneHot()
        self.greater = P.Greater()
        self.cast = P.Cast()
        self.sum_loss = P.ReduceSum()
        self.tile = P.Tile()
        self.expandims = P.ExpandDims()

        self.gather = P.GatherNd()
        self.argmax = P.ArgMaxWithValue(axis=1)

        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.value = Tensor(1.0, mstype.float16)

        self.num_bboxes = (cfg.num_expected_pos_stage2 + cfg.num_expected_neg_stage2) * batch_size

        rmv_first = np.ones((self.num_bboxes, self.num_classes))
        rmv_first[:, 0] = np.zeros((self.num_bboxes,))
        self.rmv_first_tensor = Tensor(rmv_first.astype(np.float16))

        self.num_bboxes_test = cfg.rpn_max_num * cfg.test_batch_size

    def construct(self, featuremap, bbox_targets, labels, mask):
        x_cls, x_reg = self.fpn_cls(featuremap)

        if self.training:
            bbox_weights = self.cast(self.logicaland(self.greater(labels, 0), mask), mstype.int32) * labels
            labels = self.cast(self.onehot(labels, self.num_classes, self.on_value, self.off_value), mstype.float16)
            bbox_targets = self.tile(self.expandims(bbox_targets, 1), (1, self.num_classes, 1))

            loss_cls, loss_reg = self.loss(x_cls, x_reg,
                                           bbox_targets, bbox_weights,
                                           labels,
                                           mask)
            out = (loss_cls, loss_reg)
        else:
            out = (x_cls, x_reg)

        return out

    def loss(self, cls_score, bbox_pred, bbox_targets, bbox_weights, labels, weights):
        """Loss method."""
        # loss_cls
        loss_cls, _ = self.loss_cls(cls_score, labels)
        weights = self.cast(weights, mstype.float16)
        loss_cls = loss_cls * weights
        loss_cls = self.sum_loss(loss_cls, (0,)) / self.sum_loss(weights, (0,))

        # loss_reg
        bbox_weights = self.cast(self.onehot(bbox_weights, self.num_classes, self.on_value, self.off_value),
                                 mstype.float16)
        bbox_weights = bbox_weights * self.rmv_first_tensor   #  * self.rmv_first_tensor  exclude background
        pos_bbox_pred = self.reshape(bbox_pred, (self.num_bboxes, -1, 4))
        loss_reg = self.loss_bbox(pos_bbox_pred, bbox_targets)
        loss_reg = self.sum_loss(loss_reg, (2,))
        loss_reg = loss_reg * bbox_weights
        loss_reg = loss_reg / self.sum_loss(weights, (0,))
        loss_reg = self.sum_loss(loss_reg, (0, 1))

        return loss_cls, loss_reg
