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
""" test graph fallback control flow."""
import pytest
import mindspore as ms
from mindspore import Tensor, jit, context, nn, Parameter
import numpy as np

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_if():
        x = Tensor([1])
        if x > Tensor([0]):
            while x < Tensor([7]):
                x *= 2
        return x

    res = control_flow_if()
    print(res)
    assert res == 8


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():
        x = Tensor([6]).astype("int32")
        y = Tensor([0]).astype("int32")
        if x > Tensor([0]):
            while x >= y:
                y = y + x

        return y

    res = control_flow_while()
    assert res == 12


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():
        x = Tensor([7]).astype("int32")
        y = Tensor([0]).astype("int32")
        z = np.array([1])
        if z <= 1:
            while x >= Tensor([0]).astype("int32"):
                y += x
                x = Tensor([-1]).astype("int32")
        return y

    res = control_flow_while()
    assert res == 7


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_two_cond_in_if_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():
        x = Tensor([1])
        y = Tensor([8])
        z = Tensor([12])
        if z > x + y:
            while x < y and x + y <= z:
                y = x + y + z
        x += Tensor(-1)
        return x + y

    res = control_flow_while()
    assert res == 21


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_two_cond_in_if_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():
        x = Tensor([7]).astype("int32")
        y = Tensor([0]).astype("int32")
        if x > y:
            while Tensor([0]).astype("int32") < x and y >= x - Tensor(7).astype("int32"):
                y += x
                x += Tensor([-6]).astype("int32")
        return y

    res = control_flow_while()
    assert res == 8


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_param_in_if():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor(1, ms.float32), name="name_a")

        def construct(self):
            x = Tensor(7).astype("int32")
            y = Tensor(0).astype("int32")
            if x < y:
                pass
            else:
                while x >= self.param_a:
                    y += x
                    x -= Tensor(-2).astype("int32")
                    self.param_a += y
            return self.param_a

    net = Net()
    res = net()
    assert res == 24
