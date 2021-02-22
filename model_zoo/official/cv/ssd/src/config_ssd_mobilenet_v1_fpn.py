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
#" ============================================================================

"""Config parameters for SSD models."""

from easydict import EasyDict as ed

config = ed({
    "model": "ssd_mobilenet_v1_fpn",
    "img_shape": [640, 640],
    "num_ssd_boxes": -1,
    "match_threshold": 0.5,
    "nms_threshold": 0.6,
    "min_score": 0.1,
    "max_boxes": 100,

    # learning rate settings
    "global_step": 0,
    "lr_init": 0.01333,
    "lr_end_rate": 0.0,
    "warmup_epochs": 2,
    "weight_decay": 4e-5,
    "momentum": 0.9,

    # network
    "num_default": [6, 6, 6, 6, 6],
    "extras_in_channels": [256, 512, 1024, 256, 256],
    "extras_out_channels": [256, 256, 256, 256, 256],
    "extras_strides": [1, 1, 2, 2, 2, 2],
    "extras_ratio": [0.2, 0.2, 0.2, 0.25, 0.5, 0.25],
    "feature_size": [80, 40, 20, 10, 5],
    "min_scale": 0.2,
    "max_scale": 0.95,
    "aspect_ratios": [(2, 3), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)],
    "steps": (8, 16, 32, 64, 128),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.25,
    "num_addition_layers": 4,
    "use_anchor_generator": True,
    "use_global_norm": True,

    # `mindrecord_dir` and `coco_root` are better to use absolute path.
    "feature_extractor_base_param": "/ckpt/mobilenet_v1.ckpt",
    "checkpoint_filter_list": ['network.multi_box.cls_layers.0.weight', 'network.multi_box.cls_layers.0.bias',
                               'network.multi_box.loc_layers.0.weight', 'network.multi_box.loc_layers.0.bias'],
    "mindrecord_dir": "/data/MindRecord_COCO",
    "coco_root": "/data/coco2017",
    "train_data_type": "train2017",
    "val_data_type": "val2017",
    "instances_set": "annotations/instances_{}.json",
    "classes": ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
    "voc_json": "annotations/voc_instances_val.json",
    # voc original dataset.
    "voc_root": "/data/voc_dataset",
    # if coco or voc used, `image_dir` and `anno_path` are useless.
    "image_dir": "",
    "anno_path": ""
})
