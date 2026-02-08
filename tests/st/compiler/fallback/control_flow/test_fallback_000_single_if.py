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
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
def test_single_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if():
        x = Tensor(7).astype("int32")
        y = Tensor(0).astype("int32")
        z = x + y
        if z > y:
            y = 5 * x + Tensor(7).astype("int32")
        return y
    res = control_flow_if()
    assert res == 42


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.0)
def test_single_if_else_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_else():
        x = Tensor(1)
        if x > Tensor(7):
            return x
        x += Tensor(3)
        return x * 2
    res = control_flow_if_else()
    assert res == 8


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.0)
def test_single_if_two_cond():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if():
        x = Tensor(1)
        y = np.array(2)
        if x < Tensor(7) and x < Tensor(y):
            return x
        return x * 2
    res = control_flow_if()
    assert res == 1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
def test_single_if_builtin_function_sum():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if x + y > 0:
            return sum([x, y, 2 * x])
        return x * 2
    res = control_flow_if()
    assert res == -21


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.00)
def test_single_if_builtin_function_abs():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        if abs(x) > Tensor(np.array(2)):
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -13


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.0)
def test_single_if_builtin_function_abs_min():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if():
        x = Tensor(-11, mstype.float32)
        y = Tensor(12, mstype.float32)
        if abs(x) > Tensor(np.array(2)) and min(x, y) == x + y:
            return x - Tensor(np.array(2))
        return x * 2
    res = control_flow_if()
    assert res == -22


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential', mem_peak=0.01)
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_single_if_tensor_asnumpy_as_condition():
    """
    Feature: JIT Fallback
    Description: Test PyExecute as condition.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def tensor_asnumpy_as_condition(x):
        cond = x.asnumpy()
        if cond:
            return x + 10
        return x

    x = Tensor(1.0, ms.float32)
    out = tensor_asnumpy_as_condition(x)
    assert out == 11


@arg_mark(plat_marks=['platform_gpu',], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_single_if_with_numpy_join():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for(a):
        x = np.array([1, 2, 3, 4])
        if a <= 15:
            x += 1
        else:
            x += 2
        return x
    a = Tensor(0)
    res = control_flow_for(a)
    assert (res == [2, 3, 4, 5]).all()
