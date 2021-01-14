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


def get_lr(init_lr, total_epoch, step_per_epoch,
           anneal_rate=0.5,
           anneal_interval=200000):
    """
    Learning rate generating

    Args:
        init_lr (float): Initial learning rate
        total_epoch (int): Total epoch
        step_per_epoch (int): Step per epoch
        anneal_rate (float): anneal rate
        anneal_interval (int ): anneal interval

    Returns:
        ndarray: learning rate

    """
    total_step = total_epoch * step_per_epoch
    lr_step = []
    for i in range(total_step):
        lr_step.append(init_lr * anneal_rate ** (i // anneal_interval))
    learning_rate = np.array(lr_step).astype(np.float32)
    return learning_rate
