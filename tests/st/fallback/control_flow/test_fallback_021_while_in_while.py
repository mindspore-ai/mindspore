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
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_if():
        x = Tensor([1])
        y = Tensor([3])
        while y > Tensor([0]):
            while x < Tensor([7]):
                x *= 2
            y -= 1
        return x

    res = control_flow_if()
    print(res)
    assert res == 8


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():

        x = Tensor([3]).astype("int32")
        y = Tensor([0]).astype("int32")
        while x + y < Tensor([6]):
            while x >= y:
                y += 1
        return y

    res = control_flow_while()
    assert res == 4


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_3():
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
        while z <= 2:
            while x >= Tensor([0]).astype("int32"):
                y += x
                x = Tensor([-1]).astype("int32")
            z += z
        return y

    res = control_flow_while()
    assert res == 7


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_with_two_cond_1():
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
        while z > x + y:
            while x < y and x + y <= z:
                y = x + y + z
        x += Tensor([5])
        return x + y

    res = control_flow_while()
    assert res == 27


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_with_two_cond_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():
        x = Tensor([7]).astype("int32")
        y = Tensor([0]).astype("int32")
        while x > y:
            while x > Tensor([0]).astype("int32") and y >= x - Tensor([7]).astype("int32"):
                y += x
                x += Tensor([-6]).astype("int32")
        return y

    res = control_flow_while()
    assert res == 8


@pytest.mark.level1
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_with_two_cond_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_while():
        x = Tensor([7]).astype("int32")
        y = Tensor([0]).astype("int32")
        while Tensor([0]).astype("int32") < x and y >= x - Tensor([7]).astype("int32"):
            while x > y:
                x += Tensor([-2]).astype("int32")
            y += x
        return y

    res = control_flow_while()
    assert res == -1


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_while_with_param():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="a")
            self.param_b = Parameter(Tensor([-1], ms.float32), name="b")

        def construct(self):
            x = Tensor([7]).astype("int32")
            y = Tensor([0]).astype("int32")
            while self.param_b < Tensor([1]).astype("int32"):
                self.param_b += Tensor([1]).astype("int32")
                while x >= self.param_a:
                    y += x
                    x -= Tensor([-2]).astype("int32")
                    self.param_a += y
            return self.param_a + self.param_b

    net = Net()
    res = net()
    assert res == 25
