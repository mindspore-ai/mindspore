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
"""centerface unique configs"""

class ConfigCenterface():
    """
    Config setup
    """
    flip_idx = [[0, 1], [3, 4]]
    default_resolution = [512, 512]
    heads = {'hm': 1, 'wh': 2, 'hm_offset': 2, 'landmarks': 5 * 2}
    head_conv = 64
    max_objs = 64

    rand_crop = True
    scale = 0.4
    shift = 0.1
    aug_rot = 0
    color_aug = True
    flip = 0.5
    input_res = 512 #768 #800
    output_res = 128 #192 #200
    num_classes = 1
    num_joints = 5
    reg_offset = True
    hm_hp = True
    reg_hp_offset = True
    dense_hp = False
    hm_weight = 1.0
    wh_weight = 0.1
    off_weight = 1.0
    lm_weight = 0.1
    rotate = 0

    # for test
    mean = [0.408, 0.447, 0.470]
    std = [0.289, 0.274, 0.278]
    test_scales = [0.999,]
    nms = 1
    flip_test = 0
    fix_res = True
    input_h = 832 #800
    input_w = 832 #800
    K = 200
    down_ratio = 4
    test_batch_size = 1

    master_batch_size = 8
    num_workers = 8
    not_rand_crop = False
    no_color_aug = False
