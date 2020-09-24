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
"""Config parameters for Darknet based yolov3_darknet53 models."""


class ConfigYOLOV3DarkNet53:
    """
    Config parameters for the yolov3_darknet53.
    Examples:
        ConfigYOLOV3DarkNet53()
    """
    # train_param
    # data augmentation related
    hue = 0.1
    saturation = 1.5
    value = 1.5
    jitter = 0.3

    resize_rate = 1
    multi_scale = [[320, 320],
                   [352, 352],
                   [384, 384],
                   [416, 416],
                   [448, 448],
                   [480, 480],
                   [512, 512],
                   [544, 544],
                   [576, 576],
                   [608, 608]
                   ]

    num_classes = 80
    max_box = 50

    backbone_input_shape = [32, 64, 128, 256, 512]
    backbone_shape = [64, 128, 256, 512, 1024]
    backbone_layers = [1, 2, 8, 8, 4]

    # confidence under ignore_threshold means no object when training
    ignore_threshold = 0.7

    # h->w
    anchor_scales = [(10, 13),
                     (16, 30),
                     (33, 23),
                     (30, 61),
                     (62, 45),
                     (59, 119),
                     (116, 90),
                     (156, 198),
                     (373, 326)]
    out_channel = 255

    # test_param
    test_img_shape = [416, 416]

    label_smooth = 0
    label_smooth_factor = 0.1
