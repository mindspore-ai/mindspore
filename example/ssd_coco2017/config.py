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

"""Config parameters for SSD models."""


class ConfigSSD:
    """
    Config parameters for SSD.

    Examples:
        ConfigSSD().
    """
    IMG_SHAPE = [300, 300]
    NUM_SSD_BOXES = 1917
    NEG_PRE_POSITIVE = 3
    MATCH_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.6
    MIN_SCORE = 0.05
    TOP_K = 100

    NUM_DEFAULT = [3, 6, 6, 6, 6, 6]
    EXTRAS_IN_CHANNELS = [256, 576, 1280, 512, 256, 256]
    EXTRAS_OUT_CHANNELS = [576, 1280, 512, 256, 256, 128]
    EXTRAS_STRIDES = [1, 1, 2, 2, 2, 2]
    EXTRAS_RATIO = [0.2, 0.2, 0.2, 0.25, 0.5, 0.25]
    FEATURE_SIZE = [19, 10, 5, 3, 2, 1]
    MIN_SCALE = 0.2
    MAX_SCALE = 0.95
    ASPECT_RATIOS = [(2,), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)]
    STEPS = (16, 32, 64, 100, 150, 300)
    PRIOR_SCALING = (0.1, 0.2)


    # `MINDRECORD_DIR` and `COCO_ROOT` are better to use absolute path.
    MINDRECORD_DIR = "/data/MindRecord_COCO"
    COCO_ROOT = "/data/coco2017"
    TRAIN_DATA_TYPE = "train2017"
    VAL_DATA_TYPE = "val2017"
    INSTANCES_SET = "annotations/instances_{}.json"
    COCO_CLASSES = ('background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
                    'teddy bear', 'hair drier', 'toothbrush')
    NUM_CLASSES = len(COCO_CLASSES)
