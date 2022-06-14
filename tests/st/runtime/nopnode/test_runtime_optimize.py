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

import numpy as np
import pytest
import mindspore
from mindspore import context, ops, nn, Tensor


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = ops.ReLU()
        self.add = ops.Add()

    def construct(self, input_x):
        output = self.relu(input_x)
        for _ in range(200):
            output = self.add(output, 1)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multi_actor_fusion():
    """
    Feature: Multi actor fusion.
    Description: Test the net which is non concurrent, that can trigger the function of multi actor fusion.
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.ones(2), mindspore.float32)
    net = Net()
    expect = np.array([201, 201])
    for _ in range(20):
        output = net(x)
        assert (output.asnumpy() == expect).all()
