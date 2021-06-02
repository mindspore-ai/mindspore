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
# ============================================================================
"""Config parameters for yolov5 models."""


class ConfigYOLOV5:
    """
    Config parameters for the yolov5.

    Examples:
        ConfigYOLOV5()
    """
    # train_param
    # data augmentation related
    hue = 0.015
    saturation = 1.5
    value = 0.4
    jitter = 0.3

    resize_rate = 10
    multi_scale = [[320, 320], [352, 352], [384, 384], [416, 416], [448, 448],
                   [480, 480], [512, 512], [544, 544], [576, 576], [608, 608],
                   [640, 640], [672, 672], [704, 704], [736, 736], [768, 768]]
    num_classes = 80
    max_box = 150

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
    test_img_shape = [640, 640]
