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
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, context
from mindspore import dtype as mstype
from mindspore.nn import Cell
from tests.st.compiler.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level0
@case_register.target_gpu
@case_register.target_ascend
def test_single_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        z = x + y
        if z > y:
            y = 5 * x + Tensor(7).astype("int32")
        return y
    res = control_flow_if()
    assert res == 42


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_if_two_cond():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(1)
        y = np.array(2)
        if x < Tensor(7) and x < Tensor(y):
            return x
        return x * 2
    res = control_flow_if()
    assert res == 1


@case_register.level1
@case_register.target_ascend
def test_single_if_builtin_function_abs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        if abs(x) > Tensor(np.array(2)):
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -13


@case_register.level1
@case_register.target_ascend
def test_single_if_builtin_function_abs_min():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if abs(x) > Tensor(np.array(2)) and min(x, y) == x + y:
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -22


@case_register.level0
@case_register.target_gpu
def test_single_if_no_else_type():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class FalseNet(Cell):
        def __init__(self):
            super(FalseNet, self).__init__()
            self.cond = False

        def construct(self):
            x = np.array(1)
            if self.cond:
                return type(2).mro()
            return type(x).mro()

    test_net = FalseNet()
    res = test_net()
    assert str(res) == "[<class 'numpy.ndarray'>, <class 'object'>]"


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_single_if_no_else_type_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    class TrueNet(Cell):
        def __init__(self):
            super(TrueNet, self).__init__()
            self.cond = True

        def construct(self):
            x = np.array(2)
            y = 2
            if self.cond:
                return type(y).mro()
            return type(x).mro()

    test_net = TrueNet()
    res = test_net()
    assert str(res) == "[<class 'int'>, <class 'object'>]"


@case_register.level0
@case_register.target_gpu
def test_single_if_tensor_asnumpy_as_condition():
    """
    Feature: JIT Fallback
    Description: Test PyExecute as condition.
    Expectation: No exception.
    """
    @jit
    def tensor_asnumpy_as_condition(x):
        cond = x.asnumpy()
        if cond:
            return x + 10
        return x

    x = Tensor(1.0, ms.float32)
    out = tensor_asnumpy_as_condition(x)
    assert out == 11
