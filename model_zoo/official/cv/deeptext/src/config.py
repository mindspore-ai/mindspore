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
# " :===========================================================================
"""
network config setting, will be used in train.py and eval.py
"""
from easydict import EasyDict as ed

config = ed({
    "img_width": 960,
    "img_height": 576,
    "keep_ratio": False,
    "flip_ratio": 0.0,
    "photo_ratio": 0.0,
    "expand_ratio": 0.3,

    # anchor
    "feature_shapes": [(36, 60)],
    "anchor_scales": [2, 4, 6, 8, 12],
    "anchor_ratios": [0.2, 0.5, 0.8, 1.0, 1.2, 1.5],
    "anchor_strides": [16],
    "num_anchors": 5 * 6,

    # rpn
    "rpn_in_channels": 512,
    "rpn_feat_channels": 640,
    "rpn_loss_cls_weight": 1.0,
    "rpn_loss_reg_weight": 3.0,
    "rpn_cls_out_channels": 1,
    "rpn_target_means": [0., 0., 0., 0.],
    "rpn_target_stds": [1.0, 1.0, 1.0, 1.0],

    # bbox_assign_sampler
    "neg_iou_thr": 0.3,
    "pos_iou_thr": 0.5,
    "min_pos_iou": 0.3,
    "num_bboxes": 5 * 6 * 36 * 60,
    "num_gts": 128,
    "num_expected_neg": 256,
    "num_expected_pos": 128,

    # proposal
    "activate_num_classes": 2,
    "use_sigmoid_cls": True,

    # roi_align
    "roi_layer": dict(type='RoIAlign', out_size=7, sample_num=2),

    # bbox_assign_sampler_stage2
    "neg_iou_thr_stage2": 0.2,
    "pos_iou_thr_stage2": 0.5,
    "min_pos_iou_stage2": 0.5,
    "num_bboxes_stage2": 2000,
    "use_ambigous_sample": True,
    "num_expected_pos_stage2": 128,
    "num_expected_amb_stage2": 128,
    "num_expected_neg_stage2": 640,
    "num_expected_total_stage2": 640,

    # rcnn
    "rcnn_in_channels": 512,
    "rcnn_fc_out_channels": 4096,
    "rcnn_loss_cls_weight": 1,
    "rcnn_loss_reg_weight": 1,
    "rcnn_target_means": [0., 0., 0., 0.],
    "rcnn_target_stds": [0.1, 0.1, 0.2, 0.2],

    # train proposal
    "rpn_proposal_nms_across_levels": False,
    "rpn_proposal_nms_pre": 2000,
    "rpn_proposal_nms_post": 2000,
    "rpn_proposal_max_num": 2000,
    "rpn_proposal_nms_thr": 0.7,
    "rpn_proposal_min_bbox_size": 0,

    # test proposal
    "rpn_nms_across_levels": False,
    "rpn_nms_pre": 1000,
    "rpn_nms_post": 1000,
    "rpn_max_num": 1000,
    "rpn_nms_thr": 0.7,
    "rpn_min_bbox_min_size": 0,
    "test_score_thr": 0.80,
    "test_iou_thr": 0.5,
    "test_max_per_img": 100,
    "test_batch_size": 2,

    "rpn_head_loss_type": "CrossEntropyLoss",
    "rpn_head_use_sigmoid": True,
    "rpn_head_weight": 1.0,

    # LR
    "base_lr": 0.02,
    "base_step": 982 * 8,
    "total_epoch": 70,
    "warmup_step": 50,
    "warmup_mode": "linear",
    "warmup_ratio": 1 / 3.0,
    "sgd_step": [8, 11],
    "sgd_momentum": 0.9,

    # train
    "batch_size": 2,
    "loss_scale": 1,
    "momentum": 0.91,
    "weight_decay": 1e-4,
    "epoch_size": 70,
    "save_checkpoint": True,
    "save_checkpoint_epochs": 10,
    "keep_checkpoint_max": 5,
    "save_checkpoint_path": "./",

    "mindrecord_dir": "/home/deeptext_sustech/data/mindrecord/full_ori",
    "use_coco": True,
    "coco_root": "/d0/dataset/coco2017",
    "cocotext_json": "/home/deeptext_sustech/data/cocotext.v2.json",
    "coco_train_data_type": "train2017",
    "num_classes": 3
})
