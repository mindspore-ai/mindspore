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
# ===========================================================================
"""config"""

class ModelConfig:
    """model config"""
    vocb_size = 184965
    batch_size = 1000
    emb_dim = 8
    lr_end = 1e-4
    lr_init = 0
    epsilon = 1e-8
    loss_scale = 1
    epoch_size = 30
    steps_per_epoch = 5166
    repeat_size = 1
    weight_bias_init = ['normal', 'normal']
    deep_layer_args = [[1024, 512, 128, 32, 1], "relu"]
    att_layer_args = [676, "relu"]
    keep_prob = 0.6
    ckpt_path = "./data/"
    keep_checkpoint_max = 50
    cats_dim = 26
    dense_dim = 13
