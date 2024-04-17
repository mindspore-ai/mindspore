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
import math
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.initializer import HeUniform

from .roi_extractor import RoIExtractor
from ..label_assignment import BBoxAssigner
from ..utils.box_utils import bbox2delta, delta2bbox


class RCNNBBoxTwoFCHead(nn.Cell):
    """
    fasterrcnn bbox head with Two fc layers to extract feature

    Args:
        in_channel (int): Input channel which can be derived by from_config
        out_channel (int): Output channel
        resolution (int): Resolution of input feature map, default 7
    """

    def __init__(self, in_channel=256, out_channel=1024, resolution=7):
        super(RCNNBBoxTwoFCHead, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc6 = nn.Dense(
            in_channel * resolution * resolution,
            out_channel,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.fc7 = nn.Dense(
            out_channel,
            out_channel,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.relu = nn.ReLU()

    def construct(self, rois_feat):
        b, n, _, _, _ = rois_feat.shape
        rois_feat = rois_feat.reshape(b * n, -1)
        fc6 = self.fc6(rois_feat)
        fc6 = self.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = self.relu(fc7)
        return fc7


def get_head(cfg, resolution=7):
    if cfg.name == "RCNNBBoxTwoFCHead":
        return RCNNBBoxTwoFCHead(in_channel=cfg.in_channel, out_channel=cfg.out_channel, resolution=resolution)

    raise InterruptedError(f"Not support bbox_head: {cfg.name}")


class BBoxHead(nn.Cell):
    """fasterrcnn bbox head"""

    def __init__(self, cfg, num_classes, with_mask=False):
        super(BBoxHead, self).__init__()
        self.head = get_head(cfg.head, cfg.resolution)
        self.roi_extractor = RoIExtractor(cfg.resolution, cfg.roi_extractor.featmap_strides)
        self.bbox_assigner = BBoxAssigner(
            rois_per_batch=cfg.bbox_assigner.rois_per_batch,
            bg_thresh=cfg.bbox_assigner.bg_thresh,
            fg_thresh=cfg.bbox_assigner.fg_thresh,
            fg_fraction=cfg.bbox_assigner.fg_fraction,
            num_classes=num_classes,
            with_mask=with_mask,
        )
        self.num_classes = num_classes
        self.cls_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="none")
        self.loc_loss = nn.SmoothL1Loss(reduction="none")
        self.bbox_cls = nn.Dense(
            cfg.head.out_channel,
            self.num_classes + 1,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.bbox_delta = nn.Dense(
            cfg.head.out_channel,
            4 * self.num_classes,
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.onehot = nn.OneHot(depth=self.num_classes)
        self.with_mask = with_mask

    def construct(self, feats, rois, rois_mask, gts, gt_masks=None):
        """
        feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_mask (Tensor): The number of RoIs in each image
        gts (Tensor): The ground-truth
        """
        if self.with_mask:
            gt_classes, gt_bboxes, gt_masks, fg_masks, valid_masks, select_rois, pos_rois = self.bbox_assigner(
                rois, rois_mask, gts, gt_masks
            )
        else:
            pos_rois = None
            gt_classes, gt_bboxes, fg_masks, valid_masks, select_rois = self.bbox_assigner(rois, rois_mask, gts)
        batch_size, rois_num, _ = select_rois.shape
        rois_feat = self.roi_extractor(feats, select_rois, valid_masks)
        feat = self.head(rois_feat)

        pred_cls = self.bbox_cls(feat).astype(ms.float32)
        pred_delta = self.bbox_delta(feat).astype(ms.float32)

        # bbox cls
        loss_bbox_cls = self.cls_loss(pred_cls, gt_classes.reshape(-1))
        loss_bbox_cls = ops.select(
            valid_masks.reshape((-1)).astype(ms.bool_), loss_bbox_cls, ops.zeros_like(loss_bbox_cls)
        )
        loss_bbox_cls = loss_bbox_cls.sum() / (valid_masks.astype(pred_cls.dtype).sum() + 1e-4)

        # bbox reg
        reg_target = bbox2delta(select_rois.reshape((-1, 4)), gt_bboxes.reshape((-1, 4)))
        reg_target = ops.tile(ops.expand_dims(reg_target, 1), (1, self.num_classes, 1))
        reg_target = ops.stop_gradient(reg_target)
        cond = ops.logical_and(gt_classes < self.num_classes, gt_classes >= 0)
        reg_class = ops.select(cond, gt_classes, ops.zeros_like(gt_classes)).reshape(-1)
        reg_class_weight = ops.expand_dims(self.onehot(reg_class), -1)
        reg_class_weight = ops.stop_gradient(
            reg_class_weight * fg_masks.reshape((-1, 1, 1)).astype(reg_class_weight.dtype))
        loss_bbox_reg = self.loc_loss(pred_delta.reshape(-1, self.num_classes, 4), reg_target)
        loss_bbox_reg = loss_bbox_reg * reg_class_weight
        loss_bbox_reg = loss_bbox_reg.sum() / (valid_masks.astype(pred_delta.dtype).sum() + 1e-4)
        if self.with_mask:
            mask_weights = reg_class_weight.reshape(batch_size, rois_num, self.num_classes) # B, N, 80, 1, 1
            mask_weights = mask_weights[:, :pos_rois.shape[1], :].reshape(-1, self.num_classes)
            return loss_bbox_reg, loss_bbox_cls, pos_rois, gt_masks, mask_weights, valid_masks
        return loss_bbox_reg, loss_bbox_cls

    def predict(self, feats, rois, rois_mask):
        batch_size, rois_num, _ = rois.shape
        rois_feat = self.roi_extractor(feats, rois, rois_mask)
        feat = self.head(rois_feat)
        pred_cls = self.bbox_cls(feat).astype(ms.float32)
        pred_cls = pred_cls.reshape((batch_size, rois_num, -1))
        pred_cls = ops.softmax(pred_cls, axis=-1)
        pred_delta = self.bbox_delta(feat).astype(ms.float32).reshape((batch_size, rois_num, self.num_classes, 4))
        rois = ops.tile(rois[:, :, :4].reshape((batch_size, rois_num, 1, 4)), (1, 1, self.num_classes, 1))
        # rois = rois.reshape((-1, rois.shape[-1]))[:, :4]
        pred_loc = delta2bbox(pred_delta.reshape((-1, 4)), rois.reshape((-1, 4)))  # true box xyxy
        pred_loc = pred_loc.reshape((batch_size, rois_num, self.num_classes * 4))

        pred_cls_mask = ops.tile(
            rois_mask.astype(ms.bool_).reshape(batch_size, rois_num, 1), (1, 1, pred_cls.shape[-1])
        )
        pred_cls = ops.select(pred_cls_mask, pred_cls, ops.ones_like(pred_cls) * -1).reshape(
            (batch_size, rois_num, self.num_classes + 1)
        )
        return ops.concat((pred_loc, pred_cls), axis=-1)
