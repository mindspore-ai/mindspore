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
from mindspore import Tensor, jit, context, nn
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_while_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(3):
            x += 1
            for _ in range(4):
                y += x
        y = y * 2 - x
        return y
    res = control_flow_for_in_while()
    assert res == 37


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_while_numpy_append():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_while():
        x = np.array([[1, 2, 3], [3, 4, 5], [4, 5, 6]])
        y = Tensor(0)
        z = x[..., 1]
        while max(z) != 7:
            z = np.append(z, 7)
            for _ in range(3):
                y += Tensor(sum(z))
        return y
    res = control_flow_for_in_while()
    assert res == 54


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_while_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class ForInWhileNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(7, mstype.int32), name='b')
            self.assign = P.Assign()

        def construct(self, x):
            self.assign(self.param_a, x + self.param_a)
            out = Tensor(0)
            y = sum(np.array([1, 2]))
            while y > 0:
                y = y - 1
                for _ in range(2):
                    out = self.param_a + self.param_b
                    self.assign(self.param_b, self.param_b + 1)
            return out

    input_x = Tensor(9, mstype.int32)
    net = ForInWhileNet()
    res = net(input_x)
    assert res == 26


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_while_print():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_while():
        x = Tensor(1)
        y = Tensor(0)
        while x < Tensor(3):
            x += y
            for _ in range(2):
                y += x
                print("After add", x, "res is ", y)
        return x, y
    res1, res2 = control_flow_for_in_while()
    assert res1 == 3
    assert res2 == 8


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_while_round():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_while():
        x = 3.14159
        y = 3
        res = 0
        while x != 3 and y >= 0:
            for _ in range(3):
                y -= round(x)
            res = x + y
            x = round(x)
        return Tensor(res)
    out = control_flow_for_in_while()
    assert out == -2.85841
