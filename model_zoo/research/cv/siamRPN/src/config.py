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
""" unique configs """
import numpy as np


class Config:
    """
    Config setup
    """
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size 271
    context_amount = 0.5                   # context amount
    sample_type = 'uniform'
    exem_stretch = False
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    # pairs per video
    pairs_per_video_per_epoch = 2
    frame_range_vid = 100  # frame range of choosing the instance
    frame_range_ytb = 1

    # training related
    checkpoint_path = r'./ckpt'
    pretrain_model = 'mindspore_alexnet.ckpt'
    train_path = r'./ytb_vid_filter'
    cur_epoch = 0
    max_epoches = 50
    batch_size = 32

    start_lr = 3e-2
    end_lr = 1e-7
    momentum = 0.9
    weight_decay = 0.0005
    check = True


    max_translate = 12                     # max translation of random shift
    max_stretch = 0.15                    # scale step of instance image
    total_stride = 8                       # total stride of backbone
    valid_scope = int((instance_size - exemplar_size) / total_stride / 2)
    anchor_scales = np.array([8,])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    anchor_base_size = 8
    pos_threshold = 0.6
    neg_threshold = 0.3
    pos_num = 16
    neg_num = 48

    # tracking related
    gray_ratio = 0.25
    score_size = int((instance_size - exemplar_size) / 8 + 1)
    penalty_k = 0.22
    window_influence = 0.40
    lr_box = 0.30
    min_scale = 0.1
    max_scale = 10

    #cloud train
    cloud_data_path = '/cache/data'


config = Config()
