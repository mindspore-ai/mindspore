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
"""Config parameters for Darknet based yolov4_cspdarknet53 models."""


class ConfigYOLOV4CspDarkNet53:
    """
    Config parameters for the yolov4_cspdarknet53.

    Examples:
        ConfigYOLOV4CspDarkNet53()
    """
    # train_param
    # data augmentation related
    hue = 0.1
    saturation = 1.5
    value = 1.5
    jitter = 0.3

    resize_rate = 10
    multi_scale = [[416, 416],
                   [448, 448],
                   [480, 480],
                   [512, 512],
                   [544, 544],
                   [576, 576],
                   [608, 608],
                   [640, 640],
                   [672, 672],
                   [704, 704],
                   [736, 736]
                   ]

    num_classes = 80
    max_box = 90

    backbone_input_shape = [32, 64, 128, 256, 512]
    backbone_shape = [64, 128, 256, 512, 1024]
    backbone_layers = [1, 2, 8, 8, 4]

    # confidence under ignore_threshold means no object when training
    ignore_threshold = 0.7

    # h->w
    anchor_scales = [(12, 16),
                     (19, 36),
                     (40, 28),
                     (36, 75),
                     (76, 55),
                     (72, 146),
                     (142, 110),
                     (192, 243),
                     (459, 401)]
    out_channel = 3 * (num_classes + 5)

    # test_param
    test_img_shape = [608, 608]

    # transfer training
    checkpoint_filter_list = ['feature_map.backblock0.conv6.weight', 'feature_map.backblock0.conv6.bias',
                              'feature_map.backblock1.conv6.weight', 'feature_map.backblock1.conv6.bias',
                              'feature_map.backblock2.conv6.weight', 'feature_map.backblock2.conv6.bias',
                              'feature_map.backblock3.conv6.weight', 'feature_map.backblock3.conv6.bias']
