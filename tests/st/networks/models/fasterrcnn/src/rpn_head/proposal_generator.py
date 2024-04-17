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
from mindspore import nn
from mindspore import ops

from ..utils.box_utils import delta2bbox


def nonempty(box, threshold):
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    valid = ops.logical_and((widths > threshold), (heights > threshold))
    return valid


def batch_nms(boxes, score, idxs, threshold):
    max_coordinate = boxes.max()
    offsets = idxs * (max_coordinate + 1)
    boxes_for_nms = (boxes + ops.expand_dims(offsets, 1)).astype(score.dtype)
    boxes_for_nms = ops.concat((boxes_for_nms, ops.expand_dims(score, -1)), axis=-1)
    output_boxes, output_idx, selected_mask = ops.NMSWithMask(threshold)(boxes_for_nms)
    return output_boxes, output_idx, selected_mask


class ProposalGenerator(nn.Cell):
    """
    Proposal generation module
    Args:
        pre_nms_top_n (int): Number of total bboxes to be kept per
            image before NMS. default 6000
        post_nms_top_n (int): Number of total bboxes to be kept per
            image after NMS. default 1000
        nms_thresh (float): Threshold in NMS. default 0.5
        min_size (float): Remove predicted boxes with either height or
             width < min_size. default 0.1
    """

    def __init__(self, pre_nms_top_n=12000, post_nms_top_n=2000, nms_thresh=0.5, min_size=1.0):
        super(ProposalGenerator, self).__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.min_size = min_size
        self.topk = ops.TopK()
        self.nms = ops.NMSWithMask(nms_thresh)

    def construct(self, scores, bbox_deltas, anchors, im_shape):
        N = scores[0].shape[0]
        B = anchors[0].shape[-1]
        pred_objectness_logits, pred_anchor_deltas = (), ()

        for score in scores:
            pred_objectness_logits = pred_objectness_logits + (score.transpose((0, 2, 3, 1)).reshape(N, -1),)
        for delta in bbox_deltas:
            pred_anchor_deltas = pred_anchor_deltas + (delta.transpose((0, 2, 3, 1)).reshape((N, -1, B)),)
        rpn_rois, rpn_rois_mask = self.predict_proposals(anchors, pred_objectness_logits, pred_anchor_deltas, im_shape)
        rpn_rois = ops.stop_gradient(rpn_rois)
        rpn_rois_mask = ops.stop_gradient(rpn_rois_mask)
        return rpn_rois, rpn_rois_mask

    def predict_proposals(self, anchors, pred_objectness_logits, pred_anchor_deltas, image_sizes):
        pred_proposals = self.decode_proposals(anchors, pred_anchor_deltas, image_sizes)
        return self.find_top_rpn_proposals(pred_proposals, pred_objectness_logits)

    def decode_proposals(self, anchors, pred_anchor_deltas, image_sizes):
        """decode pred_anchor_deltas to true box xyxy"""
        proposals = ()
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            N = pred_anchor_deltas_i.shape[0]
            B = anchors_i.shape[-1]
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            anchors_i = ops.tile(ops.expand_dims(anchors_i, 0), (N, 1, 1)).reshape(-1, B)
            proposals_i = delta2bbox(
                pred_anchor_deltas_i, anchors_i, weights=(1.0, 1.0, 1.0, 1.0), max_shape=image_sizes
            )
            proposals = proposals + (proposals_i.reshape(N, -1, B),)
        return proposals

    def find_top_rpn_proposals(self, proposals, pred_objectness_logits):
        """get top post_nms_top_n proposals"""
        # 1. Select top-k anchor after nms for every level and every image
        boxes = []
        for _, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
            batch_size, Hi_Wi_A, _ = proposals_i.shape
            temp_proposals = ops.concat(
                (ops.zeros((Hi_Wi_A, 2), proposals_i.dtype), ops.ones((Hi_Wi_A, 2), proposals_i.dtype)), axis=-1
            )
            batch_boxes = []
            logits_i = ops.sigmoid(logits_i)
            for b in range(batch_size):
                proposals_ib = proposals_i[b]
                valid = nonempty(proposals_ib, self.min_size)
                logits_ib = ops.select(valid, logits_i[b], ops.zeros_like(logits_i[b]))
                proposals_ib = ops.select(ops.tile(valid.reshape(-1, 1), (1, 4)), proposals_ib, temp_proposals)
                num_proposals_i = min(Hi_Wi_A, self.pre_nms_top_n)

                # select top num_proposals_i proposals
                _, idx = self.topk(logits_ib, num_proposals_i)
                boxes_for_nms = ops.concat((proposals_ib, ops.expand_dims(logits_ib, -1)), axis=-1)
                boxes_for_nms = boxes_for_nms[idx]
                nms_box, _, nms_mask = self.nms(boxes_for_nms)
                nms_box_logits = ops.select(
                    nms_mask, boxes_for_nms[:, 4], ops.zeros_like(nms_mask).astype(nms_box.dtype)
                )
                boxes_for_nms = ops.concat((boxes_for_nms[:, :4], ops.expand_dims(nms_box_logits, -1)), axis=-1)
                batch_boxes.append(boxes_for_nms)
            boxes.append(ops.stack(batch_boxes, 0))

        # 2. Concat all levels together
        boxes = ops.concat(boxes, axis=1)

        # 3. For each image choose topk results.
        proposal_boxes = []
        proposal_masks = []
        for b in range(boxes.shape[0]):
            box = boxes[b]
            nms_box_logits = box[:, 4]
            _, idx = self.topk(nms_box_logits, self.post_nms_top_n)
            box_keep = box[idx]
            nms_box, _, nms_mask = self.nms(box_keep)
            proposal_boxes.append(box_keep[:, :4])
            mask = ops.logical_and(box_keep[:, 4] > 0, nms_mask)
            proposal_masks.append(mask)

        proposal_boxes = ops.stack(proposal_boxes, 0)
        proposal_masks = ops.stack(proposal_masks, 0)
        proposal_boxes = ops.stop_gradient(proposal_boxes)
        proposal_masks = ops.stop_gradient(proposal_masks)
        # proposal_boxes shape is [self.batch_size, post_nms_top_n, 4] 4 is (x0, y0, x1, y1)
        return proposal_boxes, proposal_masks
