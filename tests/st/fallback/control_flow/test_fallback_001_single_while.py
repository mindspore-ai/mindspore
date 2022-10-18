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
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, context, Parameter
from mindspore.nn import Cell

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = Tensor(1)
        while x < Tensor(7):
            x *= 2
        return x
    res = control_flow_while()
    assert res == 8


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        while x >= y:
            y += x
        return y
    res = control_flow_while()
    assert res == 14


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        while x >= Tensor(0).astype("int32"):
            y += x
            x = Tensor(-1).astype("int32")
        return y
    res = control_flow_while()
    assert res == 7


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_two_cond_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = Tensor(1)
        y = Tensor(8)
        z = Tensor(12)
        while x < y and x + y <= z:
            y = x + y + z
        x += Tensor(3)
        return x + y
    res = control_flow_while()
    assert res == 25


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_two_cond_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        while Tensor(0).astype("int32") < x and y >= x - Tensor(7).astype("int32"):
            y += x
            x += Tensor(-6).astype("int32")
        return y
    res = control_flow_while()
    assert res == 8


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_param():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")

        def construct(self):
            x = Tensor(7).astype("int32")
            y = Tensor(0).astype("int32")
            while x >= self.param_a:
                y += x
                x -= Tensor(-2).astype("int32")
                self.param_a += y
            return self.param_a

    net = Net()
    res = net()
    assert res == 24


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_single_while_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([0, 2, 4, 6, 8])
        while (x == y).any():
            x[x == 2] = 1
        return Tensor(x)
    res = control_flow_while()
    assert (res.asnumpy() == [1, 1, 3, 4, 5]).all()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_single_while_two_cond_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_while():
        x = np.array([1, 2, 3, 4, 5])
        y = Tensor(1)
        while sum(x) > 0 and y >= 0:
            x -= 3
        return Tensor(sum(x))
    res = control_flow_while()
    assert res == 0
