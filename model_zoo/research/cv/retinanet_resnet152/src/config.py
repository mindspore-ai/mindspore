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
# " ============================================================================

"""Config parameters for retinanet models."""

from easydict import EasyDict as ed

config = ed({
    "img_shape": [640, 640],
    "num_retinanet_boxes": 76725,
    "match_thershold": 0.5,
    "softnms_sigma": 0.5,
    "nms_thershold": 0.6,
    "min_score": 0.1,
    "max_boxes": 100,

    # learing rate settings
    "global_step": 0,
    "lr_init": 1e-6,
    "lr_end_rate": 5e-3,
    "warmup_epochs1": 2,
    "warmup_epochs2": 5,
    "warmup_epochs3": 23,
    "warmup_epochs4": 60,
    "warmup_epochs5": 160,
    "momentum": 0.9,
    "weight_decay": 1.5e-4,

    # network
    "num_default": [9, 9, 9, 9, 9],
    "extras_out_channels": [256, 256, 256, 256, 256],
    "feature_size": [80, 40, 20, 10, 5],
    "aspect_ratios": [(0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0), (0.5, 1.0, 2.0)],
    "steps": (8, 16, 32, 64, 128),
    "anchor_size": (32, 64, 128, 256, 512),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.75,

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    "mindrecord_dir": "/opr/root/data/MindRecord_COCO",
    "coco_root": "/opr/root/data/",
    "train_data_type": "train2017",
    "val_data_type": "val2017",
    "instances_set": "anno/instances_{}.json",
    "coco_classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                     'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                     'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                     'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                     'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors',
                     'teddy bear', 'hair drier', 'toothbrush'),
    "num_classes": 81,
    # The annotation.json position of voc validation dataset.
    "voc_root": "",
    # voc original dataset.
    "voc_dir": "",
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    "image_dir": "",
    "anno_path": "",
    "save_checkpoint": True,
    "save_checkpoint_epochs": 5,
    "keep_checkpoint_max": 30,
    "save_checkpoint_path": "./model",
    "finish_epoch": 0,
    "checkpoint_path": "/opr/root/reretina/retinanet6405/LOG0/model/retinanet-10_916.ckpt"
})
