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
"""learning rate generator"""

import numpy as np


def get_lr(epoch_max, lr_base, steps_per_epoch, step=10, scale=0.457305051927326):
    """generate learning rate"""
    lr_list = []
    for epoch in range(epoch_max):
        for _ in range(steps_per_epoch):
            lr_list.append(lr_base * (scale ** (epoch // step)))
    return np.array(lr_list)
