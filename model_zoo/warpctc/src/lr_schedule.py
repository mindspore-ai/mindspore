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
"""Learning rate generator."""


def get_lr(epoch_size, step_size, lr_init):
    """
     generate learning rate for each step, which decays in every 10 epoch

     Args:
        epoch_size(int): total epoch number
        step_size(int): total step number in each step
        lr_init(int): initial learning rate

     Returns:
        List, learning rate array
     """
    lr = lr_init
    lrs = []
    for i in range(1, epoch_size + 1):
        if i % 10 == 0:
            lr *= 0.1
        lrs.extend([lr for _ in range(step_size)])
    return lrs
