# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np
import mindspore
from mindspore import nn, ops

mindspore.set_context(max_device_memory="0.1GB", device_target="Ascend")


class AddNet(nn.Cell):

    def __init__(self):
        super().__init__()
        self.add = ops.Add()

    def construct(self, x, y):
        return self.add(x, y)


def test_alloc_memory_fail():
    """
    Feature: The format of exception when memory alloc failed
    Description: Adding two large tensors together causes a memory exception
    Expectation: Throw exception that contains at least 2 message blocks
    """
    net = AddNet()
    x = mindspore.Tensor(np.random.randn(1024, 1024, 16).astype(np.float32))
    y = mindspore.Tensor(np.random.randn(1024, 1024, 16).astype(np.float32))
    r = net(x, y)
    print(r.shape)
