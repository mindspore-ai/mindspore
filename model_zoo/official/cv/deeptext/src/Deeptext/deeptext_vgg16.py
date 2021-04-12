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
"""Deeptext based on VGG16."""

import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.initializer import initializer
from mindspore.common.tensor import Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from .anchor_generator import AnchorGenerator
from .bbox_assign_sample_stage2 import BboxAssignSampleForRcnn
from .proposal_generator import Proposal
from .rcnn import Rcnn
from .rpn import RPN
from .vgg16 import VGG16FeatureExtraction


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shp_weight_conv = (out_channels, in_channels, kernel_size, kernel_size)

    shp_bias_conv = (out_channels,)
    weights = initializer('Normal', shape=shp_weight_conv, dtype=mstype.float32).to_tensor()
    bias_conv = initializer(0, shape=shp_bias_conv, dtype=mstype.float32).to_tensor()

    layers = []
    layers += [nn.Conv2d(in_channels, out_channels,
                         kernel_size=kernel_size, stride=stride, padding=padding,
                         pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=bias_conv)]
    return nn.SequentialCell(layers)


class Deeptext_VGG16(nn.Cell):
    """
    Deeptext_VGG16 Network.

    Note:
        backbone = vgg16

    Returns:
        Tuple, tuple of output tensor.
        rpn_loss: Scalar, Total loss of RPN subnet.
        rcnn_loss: Scalar, Total loss of RCNN subnet.
        rpn_cls_loss: Scalar, Classification loss of RPN subnet.
        rpn_reg_loss: Scalar, Regression loss of RPN subnet.
        rcnn_cls_loss: Scalar, Classification loss of RCNN subnet.
        rcnn_reg_loss: Scalar, Regression loss of RCNN subnet.

    Examples:
        net = Deeptext_VGG16()
    """

    def __init__(self, config):
        super(Deeptext_VGG16, self).__init__()
        self.train_batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_strides = config.anchor_strides
        self.target_means = tuple(config.rcnn_target_means)
        self.target_stds = tuple(config.rcnn_target_stds)

        # Anchor generator
        anchor_base_sizes = None
        self.anchor_base_sizes = list(
            self.anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        featmap_sizes = config.feature_shapes
        assert len(featmap_sizes) == len(self.anchor_generators)

        self.anchor_list = self.get_anchors(featmap_sizes)

        # Rpn and rpn loss
        self.gt_labels_stage1 = Tensor(np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8))
        self.rpn_with_loss = RPN(config,
                                 self.train_batch_size,
                                 config.rpn_in_channels,
                                 config.rpn_feat_channels,
                                 config.num_anchors,
                                 config.rpn_cls_out_channels)

        # Proposal
        self.proposal_generator = Proposal(config,
                                           self.train_batch_size,
                                           config.activate_num_classes,
                                           config.use_sigmoid_cls)
        self.proposal_generator.set_train_local(config, True)
        self.proposal_generator_test = Proposal(config,
                                                config.test_batch_size,
                                                config.activate_num_classes,
                                                config.use_sigmoid_cls)
        self.proposal_generator_test.set_train_local(config, False)

        # Assign and sampler stage two
        self.bbox_assigner_sampler_for_rcnn = BboxAssignSampleForRcnn(config, self.train_batch_size,
                                                                      config.num_bboxes_stage2, True)
        self.decode = P.BoundingBoxDecode(max_shape=(576, 960), means=self.target_means, \
                                          stds=self.target_stds)

        # Rcnn
        self.rcnn = Rcnn(config, config.rcnn_in_channels * config.roi_layer['out_size'] * config.roi_layer['out_size'],
                         self.train_batch_size, self.num_classes)

        # Op declare
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()

        self.concat = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)
        self.reshape = P.Reshape()
        self.select = P.Select()
        self.greater = P.Greater()
        self.transpose = P.Transpose()

        # Test mode
        self.test_batch_size = config.test_batch_size
        self.split = P.Split(axis=0, output_num=self.test_batch_size)
        self.split_shape = P.Split(axis=0, output_num=4)
        self.split_scores = P.Split(axis=1, output_num=self.num_classes)
        self.split_cls = P.Split(axis=0, output_num=self.num_classes - 1)
        self.tile = P.Tile()
        self.gather = P.GatherNd()

        self.rpn_max_num = config.rpn_max_num

        self.zeros_for_nms = Tensor(np.zeros((self.rpn_max_num, 3)).astype(np.float32))
        self.ones_mask = np.ones((self.rpn_max_num, 1)).astype(np.bool)
        self.zeros_mask = np.zeros((self.rpn_max_num, 1)).astype(np.bool)
        self.bbox_mask = Tensor(np.concatenate((self.ones_mask, self.zeros_mask,
                                                self.ones_mask, self.zeros_mask), axis=1))
        self.nms_pad_mask = Tensor(np.concatenate((self.ones_mask, self.ones_mask,
                                                   self.ones_mask, self.ones_mask, self.zeros_mask), axis=1))

        self.test_score_thresh = Tensor(np.ones((self.rpn_max_num, 1)).astype(np.float32) * config.test_score_thr)
        self.test_score_zeros = Tensor(np.ones((self.rpn_max_num, 1)).astype(np.float32) * 0)
        self.test_box_zeros = Tensor(np.ones((self.rpn_max_num, 4)).astype(np.float32) * -1)
        self.test_iou_thr = Tensor(np.ones((self.rpn_max_num, 1)).astype(np.float32) * config.test_iou_thr)
        self.test_max_per_img = config.test_max_per_img
        self.nms_test = P.NMSWithMask(config.test_iou_thr)
        self.softmax = P.Softmax(axis=1)
        self.logicand = P.LogicalAnd()
        self.oneslike = P.OnesLike()
        self.test_topk = P.TopK(sorted=True)
        self.test_num_proposal = self.test_batch_size * self.rpn_max_num

        # Improve speed
        self.concat_start = (self.num_classes - 2)
        self.concat_end = (self.num_classes - 1)

        # Init tensor
        self.use_ambigous_sample = config.use_ambigous_sample
        roi_align_index = [np.array(np.ones((config.num_expected_pos_stage2 + config.num_expected_neg_stage2, 1)) * i,
                                    dtype=np.float32) for i in range(self.train_batch_size)]
        if self.use_ambigous_sample:
            roi_align_index = [np.array(np.ones((
                config.num_expected_pos_stage2 + config.num_expected_amb_stage2 + config.num_expected_neg_stage2,
                1)) * i,
                                        dtype=np.float32) for i in range(self.train_batch_size)]

        roi_align_index_test = [np.array(np.ones((config.rpn_max_num, 1)) * i, dtype=np.float32) \
                                for i in range(self.test_batch_size)]

        self.roi_align_index_tensor = Tensor(np.concatenate(roi_align_index))
        self.roi_align_index_test_tensor = Tensor(np.concatenate(roi_align_index_test))

        self.roi_align4 = P.ROIAlign(pooled_width=7, pooled_height=7, spatial_scale=0.125)
        self.roi_align5 = P.ROIAlign(pooled_width=7, pooled_height=7, spatial_scale=0.0625)

        self.concat1 = P.Concat(axis=1)
        self.roi_align_fuse = _conv(in_channels=1024, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.vgg16_feature_extractor = VGG16FeatureExtraction()

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids):
        _, _, _, f4, f5 = self.vgg16_feature_extractor(img_data)
        f4 = self.cast(f4, mstype.float32)
        f5 = self.cast(f5, mstype.float32)
        x = (f4, f5)

        rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss, _ = self.rpn_with_loss(x,
                                                                                           img_metas,
                                                                                           self.anchor_list,
                                                                                           gt_bboxes,
                                                                                           self.gt_labels_stage1,
                                                                                           gt_valids)

        if self.training:
            proposal, proposal_mask = self.proposal_generator(cls_score, bbox_pred, self.anchor_list)
        else:
            proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred, self.anchor_list)

        gt_labels = self.cast(gt_labels, mstype.int32)
        gt_valids = self.cast(gt_valids, mstype.int32)
        bboxes_tuple = ()
        deltas_tuple = ()
        labels_tuple = ()
        mask_tuple = ()
        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

                bboxes, deltas, labels, mask = self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i,
                                                                                   gt_labels_i,
                                                                                   proposal_mask[i],
                                                                                   proposal[i][::, 0:4:1],
                                                                                   gt_valids_i)
                bboxes_tuple += (bboxes,)
                deltas_tuple += (deltas,)
                labels_tuple += (labels,)
                mask_tuple += (mask,)

            bbox_targets = self.concat(deltas_tuple)
            rcnn_labels = self.concat(labels_tuple)
            bbox_targets = F.stop_gradient(bbox_targets)
            rcnn_labels = F.stop_gradient(rcnn_labels)
            rcnn_labels = self.cast(rcnn_labels, mstype.int32)
        else:
            mask_tuple += proposal_mask
            bbox_targets = proposal_mask
            rcnn_labels = proposal_mask
            for p_i in proposal:
                bboxes_tuple += (p_i[::, 0:4:1],)

        if self.training:
            if self.train_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_tensor, bboxes_all))
        else:
            if self.test_batch_size > 1:
                bboxes_all = self.concat(bboxes_tuple)
            else:
                bboxes_all = bboxes_tuple[0]
            rois = self.concat_1((self.roi_align_index_test_tensor, bboxes_all))

        rois = self.cast(rois, mstype.float32)
        rois = F.stop_gradient(rois)

        roi_feats = self.roi_align5(x[1], rois)
        roi_align4_out = self.roi_align4(x[0], rois)

        roi_align4_out = self.cast(roi_align4_out, mstype.float32)
        roi_feats = self.cast(roi_feats, mstype.float32)
        roi_feats = self.concat1((roi_feats, roi_align4_out))

        roi_feats = self.cast(roi_feats, mstype.float32)
        roi_feats = self.roi_align_fuse(roi_feats)

        roi_feats = self.cast(roi_feats, mstype.float32)

        rcnn_masks = self.concat(mask_tuple)
        rcnn_masks = F.stop_gradient(rcnn_masks)
        rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))
        rcnn_loss, rcnn_cls_loss, rcnn_reg_loss, _ = self.rcnn(roi_feats,
                                                               bbox_targets,
                                                               rcnn_labels,
                                                               rcnn_mask_squeeze)

        output = ()
        if self.training:
            output += (rpn_loss, rcnn_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss)
        else:
            output = self.get_det_bboxes(rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, bboxes_all, img_metas)

        return output

    def get_det_bboxes(self, cls_logits, reg_logits, mask_logits, rois, img_metas):
        """Get the actual detection box."""
        scores = self.softmax(cls_logits)

        boxes_all = ()
        for i in range(self.num_classes):
            k = i * 4
            reg_logits_i = self.squeeze(reg_logits[::, k:k + 4:1])
            out_boxes_i = self.decode(rois, reg_logits_i)
            boxes_all += (out_boxes_i,)

        scores_all = self.split(scores)
        mask_all = self.split(self.cast(mask_logits, mstype.int32))

        boxes_all_with_batchsize = ()
        for i in range(self.test_batch_size):
            boxes_tuple = ()
            for j in range(self.num_classes):
                boxes_tmp = self.split(boxes_all[j])
                out_boxes_h = boxes_tmp[i] / 1
                out_boxes_w = boxes_tmp[i] / 1
                boxes_tuple += (self.select(self.bbox_mask, out_boxes_w, out_boxes_h),)
            boxes_all_with_batchsize += (boxes_tuple,)

        output = self.multiclass_nms(boxes_all_with_batchsize, scores_all, mask_all)

        return output

    def multiclass_nms(self, boxes_all, scores_all, mask_all):
        """Multiscale postprocessing."""
        all_bboxes = ()
        all_labels = ()
        all_masks = ()

        for i in range(self.test_batch_size):
            bboxes = boxes_all[i]
            scores = scores_all[i]
            masks = self.cast(mask_all[i], mstype.bool_)

            res_boxes_tuple = ()
            res_labels_tuple = ()
            res_masks_tuple = ()

            for j in range(self.num_classes - 1):
                k = j + 1
                _cls_scores = scores[::, k:k + 1:1]
                _bboxes = self.squeeze(bboxes[k])
                _mask_o = self.reshape(masks, (self.rpn_max_num, 1))

                cls_mask = self.greater(_cls_scores, self.test_score_thresh)
                _mask = self.logicand(_mask_o, cls_mask)

                _reg_mask = self.cast(self.tile(self.cast(_mask, mstype.int32), (1, 4)), mstype.bool_)

                _bboxes = self.select(_reg_mask, _bboxes, self.test_box_zeros)
                _cls_scores = self.select(_mask, _cls_scores, self.test_score_zeros)
                __cls_scores = self.squeeze(_cls_scores)
                scores_sorted, topk_inds = self.test_topk(__cls_scores, self.rpn_max_num)
                topk_inds = self.reshape(topk_inds, (self.rpn_max_num, 1))
                scores_sorted = self.reshape(scores_sorted, (self.rpn_max_num, 1))
                _bboxes_sorted = self.gather(_bboxes, topk_inds)
                _mask_sorted = self.gather(_mask, topk_inds)

                scores_sorted = self.tile(scores_sorted, (1, 4))
                cls_dets = self.concat_1((_bboxes_sorted, scores_sorted))
                cls_dets = P.Slice()(cls_dets, (0, 0), (self.rpn_max_num, 5))

                cls_dets, _index, _mask_nms = self.nms_test(cls_dets)
                _index = self.reshape(_index, (self.rpn_max_num, 1))
                _mask_nms = self.reshape(_mask_nms, (self.rpn_max_num, 1))

                _mask_n = self.gather(_mask_sorted, _index)

                _mask_n = self.logicand(_mask_n, _mask_nms)
                cls_labels = self.oneslike(_index) * j
                res_boxes_tuple += (cls_dets,)
                res_labels_tuple += (cls_labels,)
                res_masks_tuple += (_mask_n,)

            res_boxes_start = self.concat(res_boxes_tuple[:self.concat_start])
            res_labels_start = self.concat(res_labels_tuple[:self.concat_start])
            res_masks_start = self.concat(res_masks_tuple[:self.concat_start])

            res_boxes_end = self.concat(res_boxes_tuple[self.concat_start:self.concat_end])
            res_labels_end = self.concat(res_labels_tuple[self.concat_start:self.concat_end])
            res_masks_end = self.concat(res_masks_tuple[self.concat_start:self.concat_end])

            res_boxes = self.concat((res_boxes_start, res_boxes_end))
            res_labels = self.concat((res_labels_start, res_labels_end))
            res_masks = self.concat((res_masks_start, res_masks_end))
            reshape_size = (self.num_classes - 1) * self.rpn_max_num
            if self.use_ambigous_sample:
                res_boxes = res_boxes_tuple[0]
                res_labels = res_labels_tuple[0]
                res_masks = res_masks_tuple[0]
                reshape_size = self.rpn_max_num

            res_boxes = self.reshape(res_boxes, (1, reshape_size, 5))
            res_labels = self.reshape(res_labels, (1, reshape_size, 1))
            res_masks = self.reshape(res_masks, (1, reshape_size, 1))

            all_bboxes += (res_boxes,)
            all_labels += (res_labels,)
            all_masks += (res_masks,)

        all_bboxes = self.concat(all_bboxes)
        all_labels = self.concat(all_labels)
        all_masks = self.concat(all_masks)
        return all_bboxes, all_labels, all_masks

    def get_anchors(self, featmap_sizes):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = ()
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i])
            multi_level_anchors += (Tensor(anchors.astype(np.float32)),)

        return multi_level_anchors

class Deeptext_VGG16_Infer(nn.Cell):
    def __init__(self, config):
        super(Deeptext_VGG16_Infer, self).__init__()
        self.network = Deeptext_VGG16(config)
        self.network.set_train(False)

    def construct(self, img_data):
        output = self.network(img_data, None, None, None, None)
        return output
