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
from mindspore import context, nn, ops, Tensor, Parameter


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.print = ops.Print()
        self.param_a = Parameter(Tensor(5, mindspore.int32), name='a')
        self.param_b = Parameter(Tensor(4, mindspore.int32), name='b')

    def construct(self, x):
        out = 0
        for _ in range(2):
            out += self.func1(x)
        return out

    def func1(self, x):
        out = x
        i = 0
        while i < 1:
            out += self.func2(x)
            i = i + 1
            self.print(out)
        return out

    def func2(self, x):
        if x > 10:
            return self.param_a
        return self.param_b


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_repeat_control_arrow_for_stack_actor():
    """
    Feature: Runtime.
    Description: Duplicate side effects depend on stack actors..
    Expectation: No exception.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor(np.array([1]), mindspore.int32)
    net = Net()
    out = net(x)
    result = 10
    assert out == result
