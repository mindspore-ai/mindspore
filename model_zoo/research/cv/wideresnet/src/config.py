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
"""Config parameters for WideResNet models."""


class Config_WideResNet:
    """
    Config parameters for the WideResNet.

    Examples:
        Config_WideResNet()
    """
    num_classes = 10
    batch_size = 32
    epoch_size = 300
    save_checkpoint_path = "./"
    repeat_num = 1
    widen_factor = 10
    depth = 40
    lr_init = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    loss_scale = 32
    save_checkpoint = True
    save_checkpoint_epochs = 5
    keep_checkpoint_max = 10
    use_label_smooth = True
    label_smooth_factor = 0.1
    pretrain_epoch_size = 0
    warmup_epochs = 5


config_WideResnet = Config_WideResNet()
