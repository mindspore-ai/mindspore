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
"""Deeptext tpositive and negative sample screening for Rcnn."""

import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor


class BboxAssignSampleForRcnn(nn.Cell):
    """
    Bbox assigner and sampler definition.

    Args:
        config (dict): Config.
        batch_size (int): Batchsize.
        num_bboxes (int): The anchor nums.
        add_gt_as_proposals (bool): add gt bboxes as proposals flag.

    Returns:
        Tensor, output tensor.
        bbox_targets: bbox location, (batch_size, num_bboxes, 4)
        bbox_weights: bbox weights, (batch_size, num_bboxes, 1)
        labels: label for every bboxes, (batch_size, num_bboxes, 1)
        label_weights: label weight for every bboxes, (batch_size, num_bboxes, 1)

    Examples:
        BboxAssignSampleForRcnn(config, 2, 1024, True)
    """

    def __init__(self, config, batch_size, num_bboxes, add_gt_as_proposals):
        super(BboxAssignSampleForRcnn, self).__init__()
        cfg = config
        self.use_ambigous_sample = cfg.use_ambigous_sample
        self.batch_size = batch_size
        self.neg_iou_thr = cfg.neg_iou_thr_stage2
        self.pos_iou_thr = cfg.pos_iou_thr_stage2
        self.min_pos_iou = cfg.min_pos_iou_stage2
        self.num_gts = cfg.num_gts
        self.num_bboxes = num_bboxes
        self.num_expected_pos = cfg.num_expected_pos_stage2
        self.num_expected_amb = cfg.num_expected_amb_stage2
        self.num_expected_neg = cfg.num_expected_neg_stage2
        self.num_expected_total = cfg.num_expected_total_stage2

        self.add_gt_as_proposals = add_gt_as_proposals
        self.label_inds = Tensor(np.arange(1, self.num_gts + 1).astype(np.int32))
        self.add_gt_as_proposals_valid = Tensor(np.array(self.add_gt_as_proposals * np.ones(self.num_gts),
                                                         dtype=np.int32))

        self.concat = P.Concat(axis=0)
        self.max_gt = P.ArgMaxWithValue(axis=0)
        self.max_anchor = P.ArgMaxWithValue(axis=1)
        self.sum_inds = P.ReduceSum()
        self.iou = P.IOU()
        self.greaterequal = P.GreaterEqual()
        self.greater = P.Greater()
        self.select = P.Select()
        self.gatherND = P.GatherNd()
        self.gatherV2 = P.Gather()
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()
        self.logicaland = P.LogicalAnd()
        self.less = P.Less()
        self.random_choice_with_mask_pos = P.RandomChoiceWithMask(self.num_expected_pos)
        self.random_choice_with_mask_amb = P.RandomChoiceWithMask(self.num_expected_amb)
        self.random_choice_with_mask_neg = P.RandomChoiceWithMask(self.num_expected_neg)
        self.reshape = P.Reshape()
        self.equal = P.Equal()
        self.bounding_box_encode = P.BoundingBoxEncode(means=(0.0, 0.0, 0.0, 0.0), stds=(0.1, 0.1, 0.2, 0.2))
        self.concat_axis1 = P.Concat(axis=1)
        self.logicalnot = P.LogicalNot()
        self.tile = P.Tile()

        # Check
        self.check_gt_one = Tensor(np.array(-1 * np.ones((self.num_gts, 4)), dtype=np.float32))
        self.check_anchor_two = Tensor(np.array(-2 * np.ones((self.num_bboxes, 4)), dtype=np.float32))

        # Init tensor
        self.assigned_gt_inds = Tensor(np.array(-1 * np.ones(num_bboxes), dtype=np.int32))
        self.assigned_gt_zeros = Tensor(np.array(np.zeros(num_bboxes), dtype=np.int32))
        self.assigned_gt_ones = Tensor(np.array(np.ones(num_bboxes), dtype=np.int32))
        self.assigned_amb = Tensor(np.array(-3 * np.ones(num_bboxes), dtype=np.int32))
        self.assigned_gt_ignores = Tensor(np.array(-1 * np.ones(num_bboxes), dtype=np.int32))
        self.assigned_pos_ones = Tensor(np.array(np.ones(self.num_expected_pos), dtype=np.int32))

        self.gt_ignores = Tensor(np.array(-1 * np.ones(self.num_gts), dtype=np.int32))
        self.range_pos_size = Tensor(np.arange(self.num_expected_pos).astype(np.float32))
        self.range_amb_size = Tensor(np.arange(self.num_expected_amb).astype(np.float32))
        self.check_neg_mask = Tensor(np.array(np.ones(self.num_expected_neg - self.num_expected_pos), dtype=np.bool))
        if self.use_ambigous_sample:
            self.check_neg_mask = Tensor(
                np.array(np.ones(self.num_expected_neg - self.num_expected_pos - self.num_expected_amb), dtype=np.bool))
        check_neg_mask_ignore_end = np.array(np.ones(self.num_expected_neg), dtype=np.bool)
        check_neg_mask_ignore_end[-1] = False
        self.check_neg_mask_ignore_end = Tensor(check_neg_mask_ignore_end)
        self.bboxs_neg_mask = Tensor(np.zeros((self.num_expected_neg, 4), dtype=np.float32))

        self.bboxs_amb_mask = Tensor(np.zeros((self.num_expected_amb, 4), dtype=np.float32))
        self.labels_neg_mask = Tensor(np.array(np.zeros(self.num_expected_neg), dtype=np.uint8))
        self.labels_amb_mask = Tensor(np.array(np.zeros(self.num_expected_amb) + 2, dtype=np.uint8))

        self.reshape_shape_pos = (self.num_expected_pos, 1)
        self.reshape_shape_amb = (self.num_expected_amb, 1)
        self.reshape_shape_neg = (self.num_expected_neg, 1)

        self.scalar_zero = Tensor(0.0, dtype=mstype.float32)
        self.scalar_neg_iou_thr = Tensor(self.neg_iou_thr, dtype=mstype.float32)
        self.scalar_pos_iou_thr = Tensor(self.pos_iou_thr, dtype=mstype.float32)
        self.scalar_min_pos_iou = Tensor(self.min_pos_iou, dtype=mstype.float32)

    def construct(self, gt_bboxes_i, gt_labels_i, valid_mask, bboxes, gt_valids):
        gt_bboxes_i = self.select(self.cast(self.tile(self.reshape(self.cast(gt_valids, mstype.int32), \
                                                                   (self.num_gts, 1)), (1, 4)), mstype.bool_), \
                                  gt_bboxes_i, self.check_gt_one)
        bboxes = self.select(self.cast(self.tile(self.reshape(self.cast(valid_mask, mstype.int32), \
                                                              (self.num_bboxes, 1)), (1, 4)), mstype.bool_), \
                             bboxes, self.check_anchor_two)

        overlaps = self.iou(bboxes, gt_bboxes_i)

        max_overlaps_w_gt_index, max_overlaps_w_gt = self.max_gt(overlaps)
        _, max_overlaps_w_ac = self.max_anchor(overlaps)

        neg_sample_iou_mask = self.logicaland(self.greaterequal(max_overlaps_w_gt,
                                                                self.scalar_zero),
                                              self.less(max_overlaps_w_gt,
                                                        self.scalar_neg_iou_thr))

        assigned_gt_inds = self.assigned_gt_inds
        if self.use_ambigous_sample:
            amb_sample_iou_mask = self.logicaland(self.greaterequal(max_overlaps_w_gt,
                                                                    self.scalar_neg_iou_thr),
                                                  self.less(max_overlaps_w_gt,
                                                            self.scalar_pos_iou_thr))

            assigned_gt_inds = self.select(amb_sample_iou_mask, self.assigned_amb, self.assigned_gt_inds)
        assigned_gt_inds2 = self.select(neg_sample_iou_mask, self.assigned_gt_zeros, assigned_gt_inds)

        pos_sample_iou_mask = self.greaterequal(max_overlaps_w_gt, self.scalar_pos_iou_thr)
        assigned_gt_inds3 = self.select(pos_sample_iou_mask, \
                                        max_overlaps_w_gt_index + self.assigned_gt_ones, assigned_gt_inds2)

        for j in range(self.num_gts):
            max_overlaps_w_ac_j = max_overlaps_w_ac[j:j + 1:1]
            overlaps_w_ac_j = overlaps[j:j + 1:1, ::]
            temp1 = self.greaterequal(max_overlaps_w_ac_j, self.scalar_min_pos_iou)
            temp2 = self.squeeze(self.equal(overlaps_w_ac_j, max_overlaps_w_ac_j))
            pos_mask_j = self.logicaland(temp1, temp2)
            assigned_gt_inds3 = self.select(pos_mask_j, (j + 1) * self.assigned_gt_ones, assigned_gt_inds3)

        assigned_gt_inds5 = self.select(valid_mask, assigned_gt_inds3, self.assigned_gt_ignores)

        bboxes = self.concat((gt_bboxes_i, bboxes))
        label_inds_valid = self.select(gt_valids, self.label_inds, self.gt_ignores)
        label_inds_valid = label_inds_valid * self.add_gt_as_proposals_valid
        assigned_gt_inds5 = self.concat((label_inds_valid, assigned_gt_inds5))

        # Get pos index
        pos_index, valid_pos_index = self.random_choice_with_mask_pos(self.greater(assigned_gt_inds5, 0))

        pos_check_valid = self.cast(self.greater(assigned_gt_inds5, 0), mstype.float32)
        pos_check_valid = self.sum_inds(pos_check_valid, -1)
        valid_pos_index = self.less(self.range_pos_size, pos_check_valid)
        pos_index = pos_index * self.reshape(self.cast(valid_pos_index, mstype.int32), (self.num_expected_pos, 1))

        num_pos = self.sum_inds(self.cast(self.logicalnot(valid_pos_index), mstype.float32), -1)
        valid_pos_index = self.cast(valid_pos_index, mstype.int32)
        pos_index = self.reshape(pos_index, self.reshape_shape_pos)
        valid_pos_index = self.reshape(valid_pos_index, self.reshape_shape_pos)
        pos_index = pos_index * valid_pos_index

        pos_assigned_gt_index = self.gatherND(assigned_gt_inds5, pos_index) - self.assigned_pos_ones
        pos_assigned_gt_index = self.reshape(pos_assigned_gt_index, self.reshape_shape_pos)
        pos_assigned_gt_index = pos_assigned_gt_index * valid_pos_index

        pos_gt_labels = self.gatherND(gt_labels_i, pos_assigned_gt_index)

        # Get ambiguous index
        num_amb = None
        amb_index = None
        valid_amb_index = None
        if self.use_ambigous_sample:
            amb_index, valid_amb_index = self.random_choice_with_mask_amb(self.equal(assigned_gt_inds5, -3))

            amb_check_valid = self.cast(self.equal(assigned_gt_inds5, -3), mstype.float32)
            amb_check_valid = self.sum_inds(amb_check_valid, -1)
            valid_amb_index = self.less(self.range_amb_size, amb_check_valid)
            amb_index = amb_index * self.reshape(self.cast(valid_amb_index, mstype.int32), (self.num_expected_amb, 1))

            num_amb = self.sum_inds(self.cast(self.logicalnot(valid_amb_index), mstype.float32), -1)
            valid_amb_index = self.cast(valid_amb_index, mstype.int32)
            amb_index = self.reshape(amb_index, self.reshape_shape_amb)
            valid_amb_index = self.reshape(valid_amb_index, self.reshape_shape_amb)
            amb_index = amb_index * valid_amb_index

        # Get neg index
        neg_index, valid_neg_index = self.random_choice_with_mask_neg(self.equal(assigned_gt_inds5, 0))

        unvalid_pos_index = self.less(self.range_pos_size, num_pos)
        if self.use_ambigous_sample:
            unvalid_amb_index = self.less(self.range_amb_size, num_amb)
            valid_neg_index = self.logicaland(self.concat((self.check_neg_mask, unvalid_amb_index, unvalid_pos_index)),
                                              valid_neg_index)
        else:
            valid_neg_index = self.logicaland(self.concat((self.check_neg_mask, unvalid_pos_index)), valid_neg_index)
        valid_neg_index = self.logicaland(valid_neg_index, self.check_neg_mask_ignore_end)
        neg_index = self.reshape(neg_index, self.reshape_shape_neg)

        valid_neg_index = self.cast(valid_neg_index, mstype.int32)
        valid_neg_index = self.reshape(valid_neg_index, self.reshape_shape_neg)
        neg_index = neg_index * valid_neg_index

        pos_bboxes_ = self.gatherND(bboxes, pos_index)

        amb_bboxes_ = None
        if self.use_ambigous_sample:
            amb_bboxes_ = self.gatherND(bboxes, amb_index)

        neg_bboxes_ = self.gatherV2(bboxes, self.squeeze(neg_index), 0)
        pos_assigned_gt_index = self.reshape(pos_assigned_gt_index, self.reshape_shape_pos)
        pos_gt_bboxes_ = self.gatherND(gt_bboxes_i, pos_assigned_gt_index)
        pos_bbox_targets_ = self.bounding_box_encode(pos_bboxes_, pos_gt_bboxes_)

        total_bboxes = self.concat((pos_bboxes_, neg_bboxes_))
        total_deltas = self.concat((pos_bbox_targets_, self.bboxs_neg_mask))
        total_labels = self.concat((pos_gt_labels, self.labels_neg_mask))

        if self.use_ambigous_sample:
            total_bboxes = self.concat((pos_bboxes_, amb_bboxes_, neg_bboxes_))
            total_deltas = self.concat((pos_bbox_targets_, self.bboxs_amb_mask, self.bboxs_neg_mask))
            total_labels = self.concat((pos_gt_labels, self.labels_amb_mask, self.labels_neg_mask))

        valid_pos_index = self.reshape(valid_pos_index, self.reshape_shape_pos)
        valid_neg_index = self.reshape(valid_neg_index, self.reshape_shape_neg)
        total_mask = self.concat((valid_pos_index, valid_neg_index))
        if self.use_ambigous_sample:
            valid_amb_index = self.reshape(valid_amb_index, self.reshape_shape_amb)
            total_mask = self.concat((valid_pos_index, valid_amb_index, valid_neg_index))

        return total_bboxes, total_deltas, total_labels, total_mask
