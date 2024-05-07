# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""test python built-in functions in graph mode"""
import pytest
import numpy as np
from mindspore import Tensor, context, nn
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_fallback_abs_tensor():
    """
    Feature: JIT Fallback
    Description: Test abs(Tensor) with a variable tensor in graph mode
    Expectation: No exception
    """

    class Net(nn.Cell):
        def construct(self, y):
            x = Tensor([-1, 2])
            return abs(x + y)

    net = Net()
    assert np.all(net(Tensor([-1, 2])).asnumpy() == np.array([2, 4]))


class GetAttrNet(Cell):
    def __init__(self, attr_name, default=None):
        super().__init__()
        self.attr_name = attr_name
        self.default = default

    def construct(self, x, y):
        if (x == y).all():
            get_attr = getattr(x, self.attr_name)
        else:
            get_attr = getattr(x, self.attr_name, self.default)
        return get_attr()

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_getattr_tensor_with_default():
    """
    Feature: test tensor abs attribute.
    Description: test tensor abs attribute.
    Expectation: No exception.
    """
    x = Tensor([-1, -2, -3])
    y = Tensor([-1, -2, -3])
    net = GetAttrNet("abs", y)
    out = net(x, y)
    assert (out.asnumpy() == [1, 2, 3]).all()
