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
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_if_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = Tensor(1)
        y = Tensor(0)
        if x <= Tensor(3):
            for _ in range(4):
                y += x
                x += 1
        return y
    res = control_flow_for_in_if()
    assert res == 10


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = Tensor(1)
        y = Tensor(0)
        if y <= x + 1 and x <= Tensor(3):
            for _ in range(5):
                y += Tensor(-1)
        y = x * y
        return y
    res = control_flow_for_in_if()
    assert res == -5


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_if_param():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class ForInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

        def construct(self):
            x = np.array(10)
            if self.param_a > self.param_b:
                x = x * 2
                self.param_a += 1
                for _ in range(0, 2):
                    x = x + x
                    self.param_b += 1
            self.param_b = self.param_a + self.param_b
            return Tensor(x), self.param_b

    for_in_if_net = ForInIfNet()
    res1, res2 = for_in_if_net()
    assert res1 == 80
    assert res2 == 12


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        x = np.array([1, 1, 1])
        y = list((4, 6, -2))
        if len(y) != min(x):
            for i in range(3):
                y += x[i]
        return Tensor(y)
    out = control_flow_for_in_if()
    np.all(out.asnumpy() == np.array([7, 9, 1]))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_for_in_if_isinstance_raise():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if(x):
        if isinstance(x, Tensor):
            print("before add:", x)
            for i in range(3):
                x += Tensor(i)
                print("after add ", x)
        else:
            raise ValueError("The input is not Tensor.")
        return x
    input_x = Tensor(1)
    out = control_flow_for_in_if(input_x)
    assert out == 4


@pytest.mark.skip(reason='Not support graph fallback feature yet')
def test_for_in_if_dict_isinstance():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_for_in_if():
        dict_x = {'a': 1, 'b': 2}
        res = 0
        if isinstance(dict_x, dict):
            for key in dict_x:
                res += dict_x.get(key)
        return Tensor(res)
    out = control_flow_for_in_if()
    assert out == 3
