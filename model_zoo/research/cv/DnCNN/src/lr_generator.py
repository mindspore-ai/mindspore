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

"""computing lr"""


def step_lr(basic_lr, gamma, total_steps, data_size):
    """computing lr"""
    lr_each_step = []
    for i in range(total_steps):
        if i <= data_size * 30:
            lr = basic_lr * pow(gamma, 0)
        elif i <= data_size * 60:
            lr = basic_lr * pow(gamma, 1)
        elif i <= data_size * 90:
            lr = basic_lr * pow(gamma, 2)
        else:
            lr = basic_lr * pow(gamma, 3)
        lr_each_step.append(lr)
    return lr_each_step
