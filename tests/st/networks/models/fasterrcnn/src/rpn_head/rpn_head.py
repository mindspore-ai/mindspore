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
from mindspore import ops, nn
from mindspore.common.initializer import HeUniform
from .proposal_generator import ProposalGenerator
from .anchor_generator import AnchorGenerator
from ..label_assignment import RPNLabelAssignment


class RPNFeat(nn.Cell):
    """
    Feature extraction in RPN head

    Args:
        num_layers (int): Feat numbers
        in_channel (int): Input channel
        out_channel (int): Output channel
    """

    def __init__(self, num_layers=1, num_anchors=3, in_channel=1024, out_channel=1024):
        super(RPNFeat, self).__init__()
        self.rpn_conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            weight_init=HeUniform(math.sqrt(5)),
            has_bias=True,
            bias_init="zeros"
        )
        self.rpn_rois_score = nn.Conv2d(
            out_channel, num_anchors, 1, weight_init=HeUniform(math.sqrt(5)), has_bias=True, bias_init="zeros"
        )
        self.rpn_rois_delta = nn.Conv2d(
            out_channel, 4 * num_anchors, 1, weight_init=HeUniform(math.sqrt(5)), has_bias=True, bias_init="zeros"
        )
        self.relu = nn.ReLU()

    def construct(self, feats):
        scores = ()
        deltas = ()
        for _, feat in enumerate(feats):
            x = self.relu(self.rpn_conv(feat))
            scores = scores + (self.rpn_rois_score(x),)
            deltas = deltas + (self.rpn_rois_delta(x),)
        return scores, deltas


class RPNHead(nn.Cell):
    """
    Region Proposal Network

    Args:
        cfg(Config): rpn_head config
        backbone_feat_nums(int): backbone feature numbers
        in_channel(int): rpn feature conv in channel
        loss_rpn_bbox(Cell): bbox loss function Cell, default is MAELoss
    """

    def __init__(self, cfg, backbone_feat_nums, in_channel, loss_rpn_bbox=None):
        super(RPNHead, self).__init__()
        acfg = cfg.anchor_generator
        self.anchor_generator = AnchorGenerator(aspect_ratios=acfg.aspect_ratios,
                                                anchor_sizes=acfg.anchor_sizes,
                                                strides=acfg.strides,
                                                variance=[1.0, 1.0, 1.0, 1.0])
        self.num_anchors = self.anchor_generator.num_anchors
        self.rpn_feat = RPNFeat(backbone_feat_nums, self.num_anchors, in_channel, cfg.feat_channel)
        tr_pcfg = cfg.train_proposal
        self.train_gen_proposal = ProposalGenerator(
            min_size=tr_pcfg.min_size,
            nms_thresh=tr_pcfg.nms_thresh,
            pre_nms_top_n=tr_pcfg.pre_nms_top_n,
            post_nms_top_n=tr_pcfg.post_nms_top_n,
        )
        te_pcfg = cfg.train_proposal
        self.test_gen_proposal = ProposalGenerator(
            min_size=te_pcfg.min_size,
            nms_thresh=te_pcfg.nms_thresh,
            pre_nms_top_n=te_pcfg.pre_nms_top_n,
            post_nms_top_n=te_pcfg.post_nms_top_n,
        )
        rcfg = cfg.rpn_label_assignment
        self.rpn_target_assign = RPNLabelAssignment(
            rnp_sample_batch=rcfg.rnp_sample_batch,
            fg_fraction=rcfg.fg_fraction,
            positive_overlap=rcfg.positive_overlap,
            negative_overlap=rcfg.negative_overlap,
            use_random=rcfg.use_random,
        )
        self.loss_rpn_bbox = loss_rpn_bbox
        if self.loss_rpn_bbox is None:
            self.loss_rpn_bbox = nn.SmoothL1Loss(reduction="none")

    def construct(self, feats, gts, image_shape):
        scores, deltas = self.rpn_feat(feats)
        shapes = ()
        for feat in feats:
            shapes += (feat.shape[-2:],)
        anchors = self.anchor_generator(shapes)
        rois, rois_mask = self.train_gen_proposal(scores, deltas, anchors, image_shape)
        tgt_labels, _, tgt_deltas = self.rpn_target_assign(gts, anchors)

        # cls loss
        score_pred = ()
        batch_size = scores[0].shape[0]
        for score in scores:
            score_pred = score_pred + (ops.transpose(score, (0, 2, 3, 1)).reshape((batch_size, -1)),)
        score_pred = ops.concat(score_pred, 1)
        valid_mask = tgt_labels >= 0
        fg_mask = tgt_labels > 0

        loss_rpn_cls = ops.SigmoidCrossEntropyWithLogits()(score_pred, fg_mask.astype(score_pred.dtype))
        loss_rpn_cls = ops.select(valid_mask, loss_rpn_cls, ops.zeros_like(loss_rpn_cls))

        # reg loss
        delta_pred = ()
        for delta in deltas:
            delta_pred = delta_pred + (ops.transpose(delta, (0, 2, 3, 1)).reshape((batch_size, -1, 4)),)
        delta_pred = ops.concat(delta_pred, 1)
        loss_rpn_reg = self.loss_rpn_bbox(delta_pred, tgt_deltas)
        fg_mask = ops.tile(ops.expand_dims(fg_mask, -1), (1, 1, 4))
        loss_rpn_reg = ops.select(fg_mask, loss_rpn_reg, ops.zeros_like(loss_rpn_reg))
        loss_rpn_cls = loss_rpn_cls.sum() / (valid_mask.astype(loss_rpn_cls.dtype).sum() + 1e-4)
        loss_rpn_reg = loss_rpn_reg.sum() / (valid_mask.astype(loss_rpn_reg.dtype).sum() + 1e-4)
        return rois, rois_mask, loss_rpn_cls, loss_rpn_reg

    def predict(self, feats, image_shape):
        scores, deltas = self.rpn_feat(feats)
        shapes = ()
        for feat in feats:
            shapes += (feat.shape[-2:],)
        anchors = self.anchor_generator(shapes)
        rois, rois_mask = self.test_gen_proposal(scores, deltas, anchors, image_shape)
        return rois, rois_mask
