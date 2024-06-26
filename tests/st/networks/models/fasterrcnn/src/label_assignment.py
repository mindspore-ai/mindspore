# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import ops, nn

from .utils.box_utils import bbox2delta


class RPNLabelAssignment(nn.Cell):
    """
    RPN targets assignment module

    The assignment consists of three steps:
        1. Match anchor and ground-truth box, label the anchor with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and
           background
        3. Generate the targets for classification and regression branch
    """

    def __init__(self,
                 rnp_sample_batch=256,
                 fg_fraction=0.5,
                 positive_overlap=0.7,
                 negative_overlap=0.3,
                 ignore_thresh=-1,
                 use_random=False):
        super(RPNLabelAssignment, self).__init__()
        self.rnp_sample_batch = rnp_sample_batch
        self.fg_fraction = fg_fraction
        self.positive_overlap = positive_overlap
        self.negative_overlap = negative_overlap
        self.ignore_thresh = ignore_thresh
        self.use_random = use_random

    def construct(self, gts, anchors):
        """
        gts: ground-truth instances. [batch_size, max_gt, 5], 5 is cls_id, x, y, x, y
        anchors (Tensor): [[num_anchors_i, 4]*5], num_anchors_i are all anchors in all feature maps.
        """
        batch_size, _, _ = gts.shape
        anchors = ops.concat(anchors, 0)
        tgt_labels, tgt_bboxes, tgt_deltas = rpn_anchor_target(
            anchors,
            gts,
            self.rnp_sample_batch,
            self.positive_overlap,
            self.negative_overlap,
            self.fg_fraction,
            self.use_random,
            batch_size,
        )
        return tgt_labels, tgt_bboxes, tgt_deltas


class BBoxAssigner(nn.Cell):
    """
    fasterrcnn targets assignment module

    The assignment consists of three steps:
        1. Match RoIs and ground-truth box, label the RoIs with foreground
           or background sample
        2. Sample anchors to keep the properly ratio between foreground and
           background
        3. Generate the targets for classification and regression branch

    Args:
        rois_per_batch (int): Total number of RoIs per image.
            default 512
        fg_fraction (float): Fraction of RoIs that is labeled
            foreground, default 0.25
        fg_thresh (float): Minimum overlap required between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a foreground sample. default 0.5
        bg_thresh (float): Maximum overlap allowed between a RoI
            and ground-truth box for the (roi, gt box) pair to be
            a background sample. default 0.5
        ignore_thresh(float): Threshold for ignoring the is_crowd ground-truth
            if the value is larger than zero.
        num_classes (int): The number of class.
    """

    def __init__(self,
                 rois_per_batch=512,
                 fg_fraction=0.25,
                 fg_thresh=0.5,
                 bg_thresh=0.5,
                 ignore_thresh=-1.0,
                 num_classes=80,
                 with_mask=False):
        super(BBoxAssigner, self).__init__()
        self.rois_per_batch = rois_per_batch
        self.fg_fraction = fg_fraction
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.ignore_thresh = ignore_thresh
        self.num_classes = num_classes
        self.with_mask = with_mask

    def construct(self, rois, rois_mask, gts, masks=None):
        if self.with_mask:
            return generate_proposal_target_with_mask(
                rois,
                rois_mask,
                gts,
                masks,
                self.rois_per_batch,
                self.fg_fraction,
                self.fg_thresh,
                self.bg_thresh,
                self.num_classes,
            )
        return generate_proposal_target(
            rois,
            rois_mask,
            gts,
            self.rois_per_batch,
            self.fg_fraction,
            self.fg_thresh,
            self.bg_thresh,
            self.num_classes,
        )


def rpn_anchor_target(
        anchors,
        gt_boxes,
        rnp_sample_batch,
        rpn_positive_overlap,
        rpn_negative_overlap,
        rpn_fg_fraction,
        use_random=True,
        batch_size=1,
):
    """
    return:
    tgt_labels(Tensor): 0 or 1, indicates whether it is a positive sample
    tgt_bboxes(Tensor): matched boxes, shape is (num_samples, 5)
    tgt_deltas(Tensor): matched encoding boxes, shape is (num_samples, 4), 4 is encoding xywh.
    """
    tgt_labels = []
    tgt_bboxes = []
    tgt_deltas = []
    for i in range(batch_size):
        gt_bbox = gt_boxes[i]
        # Step1: match anchor and gt_bbox
        # matches is the matched box index of anchors
        # match_labels is the matched label of anchors, -1 is ignore label, 0 is background label.
        matches, match_labels = label_box(anchors, gt_bbox, rpn_positive_overlap, rpn_negative_overlap, True)
        # Step2: sample anchor
        fg_mask = ops.logical_and(match_labels != -1, match_labels != 0)  # nonzero
        bg_mask = match_labels == 0
        if use_random:
            fg_num = int(rnp_sample_batch * rpn_fg_fraction)
            fg_sampler = ops.RandomChoiceWithMask(count=fg_num, seed=1, seed2=1)
            fg_idx, fg_s_mask = fg_sampler(fg_mask)
            fg_mask = ops.zeros_like(fg_mask)
            fg_mask[fg_idx.reshape(-1)] = fg_s_mask

            bg_num = rnp_sample_batch - fg_num
            bg_num_mask = ms.numpy.arange(int(rnp_sample_batch)) < bg_num
            bg_sampler = ops.RandomChoiceWithMask(count=int(rnp_sample_batch), seed=1, seed2=1)
            bg_idx, bg_s_mask = bg_sampler(bg_mask)
            bg_mask = ops.zeros_like(bg_mask)
            bg_mask[bg_idx.reshape(-1)] = ops.logical_and(bg_s_mask, bg_num_mask)
        else:
            fg_num = rnp_sample_batch * rpn_fg_fraction
            fg_num = min(fg_num, fg_mask.astype(ms.float32).sum().astype(ms.int32))
            bg_num = rnp_sample_batch - fg_num
            fg_mask = ops.logical_and(ops.cumsum(fg_mask.astype(ms.float32), 0) < fg_num, fg_mask)
            bg_mask = ops.logical_and(ops.cumsum(bg_mask.astype(ms.float32), 0) < bg_num, bg_mask)

        # Fill with the ignore label (-1), then set positive and negative labels
        labels = ops.ones(match_labels.shape, ms.int32) * -1
        labels = ops.select(bg_mask, ops.zeros_like(labels), labels)
        labels = ops.select(fg_mask, ops.ones_like(labels), labels)

        # Step3: make output
        matched_gt_boxes = gt_bbox[matches]
        tgt_delta = bbox2delta(anchors, matched_gt_boxes[:, 1:], weights=(1.0, 1.0, 1.0, 1.0))
        tgt_labels.append(labels)
        tgt_bboxes.append(matched_gt_boxes)
        tgt_deltas.append(tgt_delta)
    tgt_labels = ops.stop_gradient(ops.stack(tgt_labels, 0))
    tgt_bboxes = ops.stop_gradient(ops.stack(tgt_bboxes, 0))
    tgt_deltas = ops.stop_gradient(ops.stack(tgt_deltas, 0))
    # tgt_labels:
    return tgt_labels, tgt_bboxes, tgt_deltas


# TODO mask
def label_box(anchors, gt_boxes, positive_overlap, negative_overlap, allow_low_quality):
    iou = ops.iou(anchors, gt_boxes[:, 1:])
    # when invalid gt, iou is -1
    iou = ops.select(ops.tile(gt_boxes[:, 0:1] >= 0, (1, anchors.shape[0])), iou, -ops.ones_like(iou))

    # select best matched gt per anchor
    matches, matched_vals = ops.ArgMaxWithValue(axis=0, keep_dims=False)(iou)

    # set ignored anchor with match_labels = -1
    match_labels = ops.ones(matches.shape, ms.int32) * -1

    # ignored is -1, positive is 1, negative is 0
    neg_cond = ops.logical_and(matched_vals >= 0, matched_vals < negative_overlap)
    match_labels = ops.select(neg_cond, ops.zeros_like(match_labels), match_labels)
    match_labels = ops.select(matched_vals >= positive_overlap, ops.ones_like(match_labels), match_labels)

    if allow_low_quality:
        highest_quality_foreach_gt = ops.ReduceMax(True)(iou, 1)
        pred_inds_with_highest_quality = (
            ops.logical_and(iou > 0, iou == highest_quality_foreach_gt).astype(ms.float32).sum(axis=0, keepdims=False)
        )
        match_labels = ops.select(pred_inds_with_highest_quality > 0, ops.ones_like(match_labels), match_labels)

    match_labels = match_labels.reshape((-1,))
    return matches, match_labels


def generate_proposal_target(rois, rois_mask, gts, rois_per_batch, fg_fraction, fg_thresh, bg_thresh, num_classes):
    gt_classes, gt_bboxes, valid_rois, fg_masks, valid_masks = [], [], [], [], []
    batch_size = len(rois)
    for i in range(batch_size):
        roi = rois[i]
        gt = gts[i]
        roi_mask = rois_mask[i]

        # Step1: label bbox
        # matches is the matched box index of roi
        # match_labels is the matched label of roi, -1 is ignore label, 0 is background label.
        roi = ops.concat((roi, gt[:, 1:]), 0)
        roi_mask = ops.concat((roi_mask, gt[:, 0] >= 0), 0)
        matches, match_labels = label_box(roi, gt, fg_thresh, bg_thresh, False)
        match_labels = ops.select(roi_mask.astype(ms.bool_), match_labels, ops.ones_like(match_labels) * -1)

        # Step2: sample bbox
        # structure gt_classes
        gt_class = gt[:, 0][matches].astype(ms.int32)
        gt_class = ops.select(match_labels == 0, ops.ones_like(gt_class) * num_classes, gt_class)
        gt_class = ops.select(match_labels == -1, ops.ones_like(gt_class) * -1, gt_class)

        # structure gt_box
        fg_mask = ops.logical_and(gt_class > -1, gt_class != num_classes)  # nonzero
        fg_num = int(rois_per_batch * fg_fraction)
        fg_sampler = ops.RandomChoiceWithMask(count=fg_num, seed=1, seed2=1)
        fg_idx, fg_s_mask = fg_sampler(fg_mask)

        bg_mask = gt_class == num_classes
        bg_sampler = ops.RandomChoiceWithMask(count=int(rois_per_batch), seed=1, seed2=1)
        bg_idx, bg_s_mask = bg_sampler(bg_mask)
        bg_num = int(rois_per_batch - fg_num)
        bg_num_mask = ms.numpy.arange(int(rois_per_batch)) < bg_num
        bg_s_mask = ops.logical_and(bg_s_mask, bg_num_mask)

        vaild_idx = ops.concat((fg_idx, bg_idx), 0).reshape(-1)
        vaild_mask = ops.concat((fg_s_mask, bg_s_mask), 0).reshape(-1)
        fg_s_mask = ops.concat((fg_s_mask, ops.zeros_like(bg_s_mask)), 0).reshape(-1)

        # Step3: get result
        # set ignore cls to 0
        gt_class = gt_class[vaild_idx]
        gt_class = ops.select(vaild_mask, gt_class, ops.zeros_like(gt_class))
        gt_classes.append(gt_class)
        gt_bboxes.append(gt[:, 1:][matches][vaild_idx])
        fg_masks.append(fg_s_mask)
        valid_masks.append(vaild_mask)
        valid_rois.append(roi[vaild_idx])

    gt_classes = ops.stop_gradient(ops.stack(gt_classes, 0))
    gt_bboxes = ops.stop_gradient(ops.stack(gt_bboxes, 0))
    fg_masks = ops.stop_gradient(ops.stack(fg_masks, 0))
    valid_masks = ops.stop_gradient(ops.stack(valid_masks, 0))
    valid_rois = ops.stop_gradient(ops.stack(valid_rois, 0))
    return gt_classes, gt_bboxes, fg_masks, valid_masks, valid_rois


def generate_proposal_target_with_mask(
        rois,
        rois_mask,
        gts,
        masks,
        rois_per_batch,
        fg_fraction,
        fg_thresh,
        bg_thresh,
        num_classes,
):
    gt_classes, gt_bboxes, valid_rois, pos_rois, fg_masks, valid_masks, gt_masks = [], [], [], [], [], [], []
    batch_size = len(rois)
    for i in range(batch_size):
        roi = rois[i]
        gt = gts[i]
        roi_mask = rois_mask[i]
        mask = masks[i]
        # Step1: label bbox
        # matches is the matched box index of roi
        # match_labels is the matched label of roi, -1 is ignore label, 0 is background label.
        roi = ops.concat((roi, gt[:, 1:]), 0)
        roi_mask = ops.concat((roi_mask, gt[:, 0] >= 0), 0)
        matches, match_labels = label_box(roi, gt, fg_thresh, bg_thresh, False)
        match_labels = ops.select(roi_mask.astype(ms.bool_), match_labels, ops.ones_like(match_labels) * -1)

        # Step2: sample bbox
        # structure gt_classes
        gt_class = gt[:, 0][matches].astype(ms.int32)
        gt_class = ops.select(match_labels == 0, ops.ones_like(gt_class) * num_classes, gt_class)
        gt_class = ops.select(match_labels == -1, ops.ones_like(gt_class) * -1, gt_class)

        # structure gt_box
        # structure gt_box
        fg_mask = ops.logical_and(gt_class > -1, gt_class != num_classes)  # nonzero
        fg_num = int(rois_per_batch * fg_fraction)
        fg_sampler = ops.RandomChoiceWithMask(count=fg_num, seed=1, seed2=1)
        fg_idx, fg_s_mask = fg_sampler(fg_mask)

        bg_mask = gt_class == num_classes
        bg_sampler = ops.RandomChoiceWithMask(count=int(rois_per_batch), seed=1, seed2=1)
        bg_idx, bg_s_mask = bg_sampler(bg_mask)
        bg_num = int(rois_per_batch - fg_num)
        bg_num_mask = ops.arange(int(rois_per_batch)) < bg_num
        bg_s_mask = ops.logical_and(bg_s_mask, bg_num_mask)

        fg_idx = fg_idx.reshape(-1)
        bg_idx = bg_idx.reshape(-1)
        vaild_idx = ops.concat((fg_idx, bg_idx), 0)
        vaild_mask = ops.concat((fg_s_mask, bg_s_mask), 0).reshape(-1)
        fg_s_mask = ops.concat((fg_s_mask, ops.zeros_like(bg_s_mask)), 0).reshape(-1)

        # Step3: get result
        # set ignore cls to 0
        gt_class = gt_class[vaild_idx]
        gt_class = ops.select(vaild_mask, gt_class, ops.zeros_like(gt_class))
        gt_classes.append(gt_class)
        gt_bboxes.append(gt[:, 1:][matches][vaild_idx])
        gt_masks.append(mask[matches][fg_idx])
        fg_masks.append(fg_s_mask)
        valid_masks.append(vaild_mask)
        valid_rois.append(roi[vaild_idx])
        pos_rois.append(roi[fg_idx])

    gt_classes = ops.stop_gradient(ops.stack(gt_classes, 0))
    gt_bboxes = ops.stop_gradient(ops.stack(gt_bboxes, 0))
    fg_masks = ops.stop_gradient(ops.stack(fg_masks, 0))
    gt_masks = ops.stop_gradient(ops.stack(gt_masks, 0))
    valid_masks = ops.stop_gradient(ops.stack(valid_masks, 0))
    valid_rois = ops.stop_gradient(ops.stack(valid_rois, 0))
    pos_rois = ops.stop_gradient(ops.stack(pos_rois, 0))
    return gt_classes, gt_bboxes, gt_masks, fg_masks, valid_masks, valid_rois, pos_rois
