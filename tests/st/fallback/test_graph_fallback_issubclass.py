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
""" test graph fallback buildin python function issubclass"""
import pytest
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_issubclass_list():
    """
    Feature: JIT Fallback
    Description: Test issubclass() in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.input = [1, 2, 3]

        def construct(self, x):
            if issubclass(type(self.input), (tuple, list)):
                print("The input: {} is tuple or list.".format(self.input))
            else:
                raise TypeError("The input is not tuple and list.")
            return self.input + x

    net = Net()
    x = Tensor([1])
    res = net(x)
    print("res:", res)
    assert (res.asnumpy() == [2, 3, 4]).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_issubclass_tensor():
    """
    Feature: JIT Fallback
    Description: Test issubclass() in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param = Parameter(Tensor([1, 2, 3]), name="param_a")

        def construct(self):
            if issubclass(type(self.param), (Tensor, Parameter)):
                print("The param is a Tensor.")
                self.param *= 2
            return self.param

    net = Net()
    res = net()
    print("res:", res)
    assert (res.asnumpy() == [2, 4, 6]).all()
