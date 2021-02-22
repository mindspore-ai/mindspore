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
"""RPN for fasterRCNN"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.ops import functional as F
from src.CTPN.bbox_assign_sample import BboxAssignSample

class RpnRegClsBlock(nn.Cell):
    """
       Rpn reg cls block for rpn layer

       Args:
           config(EasyDict) - Network construction config.
           in_channels (int) - Input channels of shared convolution.
           feat_channels (int) - Output channels of shared convolution.
           num_anchors (int) - The anchor number.
           cls_out_channels (int) - Output channels of classification convolution.

       Returns:
           Tensor, output tensor.
       """

    def __init__(self,
                 config,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 cls_out_channels):
        super(RpnRegClsBlock, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.shape = (-1, 2*config.hidden_size)
        self.lstm_fc = nn.Dense(2*config.hidden_size, 512).to_float(mstype.float16)
        self.rpn_cls = nn.Dense(in_channels=512, out_channels=num_anchors * cls_out_channels).to_float(mstype.float16)
        self.rpn_reg = nn.Dense(in_channels=512, out_channels=num_anchors * 4).to_float(mstype.float16)
        self.shape1 = (-1, config.num_step, config.rnn_batch_size)
        self.shape2 = (config.batch_size, -1, config.rnn_batch_size, config.num_step)
        self.transpose = P.Transpose()
        self.print = P.Print()

    def construct(self, x):
        x = self.reshape(x, self.shape)
        x = self.lstm_fc(x)
        x1 = self.rpn_cls(x)
        x1 = self.transpose(x1, (1, 0))
        x1 = self.reshape(x1, self.shape1)
        x1 = self.transpose(x1, (0, 2, 1))
        x1 = self.reshape(x1, self.shape2)
        x2 = self.rpn_reg(x)
        x2 = self.transpose(x2, (1, 0))
        x2 = self.reshape(x2, self.shape1)
        x2 = self.transpose(x2, (0, 2, 1))
        x2 = self.reshape(x2, self.shape2)
        return x1, x2

class RPN(nn.Cell):
    """
    ROI proposal network..

    Args:
        config (dict) - Config.
        batch_size (int) - Batchsize.
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        RPN(config=config, batch_size=2, in_channels=256, feat_channels=1024,
            num_anchors=3, cls_out_channels=512)
    """
    def __init__(self,
                 config,
                 batch_size,
                 in_channels,
                 feat_channels,
                 num_anchors,
                 cls_out_channels):
        super(RPN, self).__init__()
        cfg_rpn = config
        self.cfg = config
        self.num_bboxes = cfg_rpn.num_bboxes
        self.feature_anchor_shape = cfg_rpn.feature_shapes
        self.feature_anchor_shape = self.feature_anchor_shape[0] * \
            self.feature_anchor_shape[1] * num_anchors * batch_size
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.test_batch_size = cfg_rpn.test_batch_size
        self.num_layers = 1
        self.real_ratio = Tensor(np.ones((1, 1)).astype(np.float16))
        self.use_sigmoid_cls = config.use_sigmoid_cls
        if config.use_sigmoid_cls:
            self.reshape_shape_cls = (-1,)
            self.loss_cls = P.SigmoidCrossEntropyWithLogits()
            cls_out_channels = 1
        else:
            self.reshape_shape_cls = (-1, cls_out_channels)
            self.loss_cls = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.rpn_convs_list = self._make_rpn_layer(self.num_layers, in_channels, feat_channels,\
            num_anchors, cls_out_channels)

        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.concat = P.Concat(axis=0)
        self.fill = P.Fill()
        self.placeh1 = Tensor(np.ones((1,)).astype(np.float16))

        self.trans_shape = (0, 2, 3, 1)

        self.reshape_shape_reg = (-1, 4)
        self.softmax = nn.Softmax()
        self.rpn_loss_reg_weight = Tensor(np.array(cfg_rpn.rpn_loss_reg_weight).astype(np.float16))
        self.rpn_loss_cls_weight = Tensor(np.array(cfg_rpn.rpn_loss_cls_weight).astype(np.float16))
        self.num_expected_total = Tensor(np.array(cfg_rpn.num_expected_neg * self.batch_size).astype(np.float16))
        self.num_bboxes = cfg_rpn.num_bboxes
        self.get_targets = BboxAssignSample(cfg_rpn, self.batch_size, self.num_bboxes, False)
        self.CheckValid = P.CheckValid()
        self.sum_loss = P.ReduceSum()
        self.loss_bbox = P.SmoothL1Loss(beta=1.0/9.0)
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()
        self.tile = P.Tile()
        self.zeros_like = P.ZerosLike()
        self.loss = Tensor(np.zeros((1,)).astype(np.float16))
        self.clsloss = Tensor(np.zeros((1,)).astype(np.float16))
        self.regloss = Tensor(np.zeros((1,)).astype(np.float16))
        self.print = P.Print()

    def _make_rpn_layer(self, num_layers, in_channels, feat_channels, num_anchors, cls_out_channels):
        """
        make rpn layer for rpn proposal network

        Args:
        num_layers (int) - layer num.
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.

        Returns:
        List, list of RpnRegClsBlock cells.
        """
        rpn_layer = RpnRegClsBlock(self.cfg, in_channels, feat_channels, num_anchors, cls_out_channels)
        return rpn_layer

    def construct(self, inputs, img_metas, anchor_list, gt_bboxes, gt_labels, gt_valids):
        '''
        inputs(Tensor): Inputs tensor from lstm.
        img_metas(Tensor): Image shape.
        anchor_list(Tensor): Total anchor list.
        gt_labels(Tensor): Ground truth labels.
        gt_valids(Tensor): Whether ground truth is valid.
        '''
        rpn_cls_score_ori, rpn_bbox_pred_ori = self.rpn_convs_list(inputs)
        rpn_cls_score = self.transpose(rpn_cls_score_ori, self.trans_shape)
        rpn_cls_score = self.reshape(rpn_cls_score, self.reshape_shape_cls)
        rpn_bbox_pred = self.transpose(rpn_bbox_pred_ori, self.trans_shape)
        rpn_bbox_pred = self.reshape(rpn_bbox_pred, self.reshape_shape_reg)
        output = ()
        bbox_targets = ()
        bbox_weights = ()
        labels = ()
        label_weights = ()
        if self.training:
            for i in range(self.batch_size):
                valid_flag_list = self.cast(self.CheckValid(anchor_list, self.squeeze(img_metas[i:i + 1:1, ::])),\
                    mstype.int32)
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                bbox_target, bbox_weight, label, label_weight = self.get_targets(gt_bboxes_i,
                                                                                 gt_labels_i,
                                                                                 self.cast(valid_flag_list,
                                                                                           mstype.bool_),
                                                                                 anchor_list, gt_valids_i)
                bbox_weight = self.cast(bbox_weight, mstype.float16)
                label_weight = self.cast(label_weight, mstype.float16)
                bbox_targets += (bbox_target,)
                bbox_weights += (bbox_weight,)
                labels += (label,)
                label_weights += (label_weight,)
            bbox_target_with_batchsize = self.concat(bbox_targets)
            bbox_weight_with_batchsize = self.concat(bbox_weights)
            label_with_batchsize = self.concat(labels)
            label_weight_with_batchsize = self.concat(label_weights)

            bbox_target_ = F.stop_gradient(bbox_target_with_batchsize)
            bbox_weight_ = F.stop_gradient(bbox_weight_with_batchsize)
            label_ = F.stop_gradient(label_with_batchsize)
            label_weight_ = F.stop_gradient(label_weight_with_batchsize)
            rpn_cls_score = self.cast(rpn_cls_score, mstype.float32)
            if self.use_sigmoid_cls:
                label_ = self.cast(label_, mstype.float32)
            loss_cls = self.loss_cls(rpn_cls_score, label_)
            loss_cls = loss_cls * label_weight_
            loss_cls = self.sum_loss(loss_cls, (0,)) / self.num_expected_total
            rpn_bbox_pred = self.cast(rpn_bbox_pred, mstype.float32)
            bbox_target_ = self.cast(bbox_target_, mstype.float32)
            loss_reg = self.loss_bbox(rpn_bbox_pred, bbox_target_)
            bbox_weight_ = self.tile(self.reshape(bbox_weight_, (self.feature_anchor_shape, 1)), (1, 4))
            loss_reg = loss_reg * bbox_weight_
            loss_reg = self.sum_loss(loss_reg, (1,))
            loss_reg = self.sum_loss(loss_reg, (0,)) / self.num_expected_total
            loss_total = self.rpn_loss_cls_weight * loss_cls + self.rpn_loss_reg_weight * loss_reg
            output = (loss_total, rpn_cls_score_ori, rpn_bbox_pred_ori, loss_cls, loss_reg)
        else:
            output = (self.placeh1, rpn_cls_score_ori, rpn_bbox_pred_ori, self.placeh1, self.placeh1)
        return output
