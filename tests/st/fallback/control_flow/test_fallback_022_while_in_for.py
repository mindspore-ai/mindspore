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
def test_while_in_for_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_for():
        x = Tensor([7]).astype("int32")
        y = Tensor([0]).astype("int32")
        z = Tensor([0]).astype("int32")
        for _ in range(3):
            y += x
            while z < 3:
                z = z + 1
                y += z
        return y

    res = control_flow_for()
    assert res == 27


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_for_zip():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_for():
        tuple_x = (Tensor(1).astype("int32"), Tensor(3).astype("int32"), Tensor(5).astype("int32"))
        sum_x = Tensor([0]).astype("int32")
        for x in zip(tuple_x):
            sum_x += x
            y = Tensor([0]).astype("int32")
            # pylint: disable=chained-comparison
            while y < Tensor([3]).astype("int32") and y > Tensor([-1]).astype("int32"):
                y += 1
                sum_x += 1
        return sum_x

    res = control_flow_for()
    assert res == 18


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_for_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit
    def control_flow_for():
        x = np.array([1, 3, 5])
        y = np.array([0, 2, 4])
        z = Tensor(0).astype("int32")
        for _ in range(3):
            x = x + y
            while z < Tensor([10]).astype("int32"):
                z = z + Tensor(1).astype("int32")
        return Tensor(x), z

    res1, res2 = control_flow_for()
    assert (res1.asnumpy() == [1, 9, 17]).all()
    assert res2 == 10


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_while_in_for_builtin_function():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_a = Parameter(Tensor([1], ms.float32), name="name_a")

        def construct(self):
            x = Tensor(np.array(1.1)).astype("float32")
            for _ in range(3):
                while self.param_a < Tensor([7]).astype("int32"):
                    x = x + Tensor(int(0) + abs(-1))
                    self.param_a += Tensor([1]).astype("int32")
                self.param_a -= Tensor([5]).astype("int32")
            return x

    net = Net()
    res = net()
    assert res == 17.1
