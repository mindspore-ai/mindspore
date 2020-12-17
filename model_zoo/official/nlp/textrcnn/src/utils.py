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
"""training utils"""
from mindspore import dtype as mstype
from mindspore.nn.dynamic_lr import exponential_decay_lr
from mindspore import Tensor


def get_lr(cfg, dataset_size):
    if cfg.cell == "lstm":
        lr = exponential_decay_lr(cfg.lstm_base_lr, cfg.lstm_decay_rate, dataset_size * cfg.num_epochs,
                                  dataset_size,
                                  cfg.lstm_decay_epoch)
        lr_ret = Tensor(lr, mstype.float32)
    else:
        lr_ret = cfg.lr
    return lr_ret
