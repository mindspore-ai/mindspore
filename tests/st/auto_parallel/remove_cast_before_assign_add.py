# Copyright 2024 Huawei Technologies Co., Ltd
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

"""Distributed Operator Parallel Example"""

import numpy as np
import mindspore as ms
from mindspore import Tensor, ops, nn
from mindspore.communication import init
from mindspore.common.initializer import initializer

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_context(max_device_memory="28GB")
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL)
ms.set_auto_parallel_context(full_batch=True)

init()
ms.set_seed(1)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.AssignAdd = ops.AssignAdd()
        self.variable = ms.Parameter(initializer(1, [1], ms.float32), name="global_step")

    def construct(self, x):
        self.AssignAdd(self.variable, x)
        return self.variable


def test_remove_cast_before_assign_add():
    """
    Feature: remove_cast_before_assign_add run semi_auto_parallel
    Description: Test remove_cast_before_assign_add feature.
    Expectation: Run success.
    """
    net = Net()
    value = Tensor(np.ones([1]).astype(np.float16) * 100)
    output = net(value)
    print(output)
    print(net.variable.asnumpy())
