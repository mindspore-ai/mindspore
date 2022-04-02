# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test high order control flow """
import pytest
from mindspore.nn import Cell
from mindspore.common import Tensor, dtype
import mindspore.ops.functional as F


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_high_control_while():
    """
    Feature: High-order differential function.
    Description: Infer of the high-order differential function.
    Expectation: Null.
    """

    class Net(Cell):
        def construct(self, x):
            while x < 10:
                x = (x * 2)
            return x

    net = Net()
    x = Tensor(1, dtype.float32)
    grad_net = F.grad(net)
    order_grad_net = F.grad(grad_net)
    order_grad = order_grad_net(x)
    assert order_grad == 0.0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_high_control_for_while():
    """
    Feature: High-order differential function.
    Description: Infer of the complex high-order differential function.
    Expectation: Null.
    """

    class Net(Cell):
        def construct(self, x):
            for _ in [2]:
                for _ in [2]:
                    while x > 1:
                        x = (x / 3)
                        x = (x / 2)
                for _ in [2]:
                    x = (x / 1)
                    x = (x + 1)
            for _ in [3]:
                for _ in [4]:
                    x = (x / 1)
                    x = (x + 3)
                for _ in [5]:
                    x = (x / 3)
                    x = (x / 2)
            return x

    net = Net()
    x = Tensor(4, dtype.float32)
    grad_net = F.grad(net)
    grad_grad_net = F.grad(grad_net)
    result = grad_grad_net(x)
    assert result == 0.0
