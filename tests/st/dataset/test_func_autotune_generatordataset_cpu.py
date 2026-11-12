# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""Dataset regression coverage for func autotune generatordataset cpu."""
import os
import time
import numpy as np
import mindspore.nn as nn
from mindspore import Parameter
from mindspore import Tensor
from mindspore import context
from mindspore.train.callback import Callback

class Net(nn.Cell):
    """simple network"""

    def __init__(self):
        super().__init__()
        self.w = Parameter(Tensor(np.ones([1]).astype(np.int32)), name='w')

    def construct(self, *dataset):
        '''calculation '''
        return dataset[0] + self.w


class TimeMonitor(Callback):
    """Time Monitor."""

    def __init__(self, dataset_size):
        super().__init__()
        self.dataset_size = dataset_size
        self.epoch_mseconds = []
        self.per_step_mseconds = []

    def epoch_begin(self, run_context):
        '''begin time '''
        self.epoch_time = time.time()
        print(run_context)

    def epoch_end(self, run_context):
        '''end time '''
        epoch_msecond = (time.time() - self.epoch_time) * 1000
        self.epoch_mseconds.append(epoch_msecond)
        per_step_msecond = epoch_msecond / self.dataset_size
        self.per_step_mseconds.append(per_step_msecond)
        print("Epoch time: {:5.3f}, per step time: {:5.3f}".format(epoch_msecond, per_step_msecond), flush=True)
        print(run_context)


def _set_context(train_mode, device, distribute):
    """
    setting context
    :param train_mode: train mode, context.GRAPH_MODE or context.PYNATIVE_MODE
    :param device: device where the code will be run, Ascend or GPU or CPU
    :param distribute: run distribute or not, True or False
    :return: None
    """
    # set context
    device_id = int(os.getenv('DEVICE_ID')) if os.getenv('DEVICE_ID') else 0
    context.set_context(mode=train_mode, device_target=device)
    context.set_context(device_id=device_id)

class MyData():
    '''test autotune dataset '''

    @staticmethod
    def __getitem__(item):
        return np.random.randn(10, 10, 3).astype(np.float32), np.array([1]).astype(np.float32), np.array([10]).astype(
            np.float32)

    @staticmethod
    def __len__():
        return 100


def func_test(x, y):
    '''
    used by map
    '''
    return x + y
