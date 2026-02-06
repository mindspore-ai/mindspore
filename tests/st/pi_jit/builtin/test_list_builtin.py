# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest 
import numpy as np
from mindspore import Tensor, jit, context, ops
from mindspore.nn import Cell
from ..share.utils import match_array, assert_executed_by_graph_mode
from ..share.grad import GradOfFirstInput
from tests.mark_utils import arg_mark


@jit(capture_mode="bytecode")
def fallback_list_with_input_tuple(a):
    res = list(a)
    return res


@jit(capture_mode="bytecode")
def fallback_list_with_input_dict(a):
    res = list(a)
    return res


@jit(capture_mode="bytecode")
def fallback_list_with_input_numpy_array(a):
    res = list(a)
    return res


@jit(capture_mode="bytecode")
def fallback_list_with_input_numpy_tensor(a, b):
    res = list(a)
    res2 = list(b)
    res3 = list(())
    return res, res2, res3


@jit
def ms_fallback_list_with_input_tuple(a):
    res = list(a)
    return res


@jit
def ms_fallback_list_with_input_dict(a):
    res = list(a)
    return res


@jit
def ms_fallback_list_with_input_numpy_array():
    a = np.array([1, 2, 3])
    res = list(a)
    return res


@jit
def ms_fallback_list_with_input_numpy_tensor(a, b):
    res = list(a)
    res2 = list(b)
    res3 = list(())
    return res, res2, res3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_list_with_input_tuple])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_tuple])
@pytest.mark.parametrize('a', [(1, 2, 3)])
def test_list_with_input_tuple(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: tuple'''
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_list_with_input_dict])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_dict])
@pytest.mark.parametrize('a', [{'a': 1, 'b': 2, 'c': 3}])
def test_list_with_input_dict(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: dict'''
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_list_with_input_numpy_array])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_numpy_array])
@pytest.mark.parametrize('a', [np.array([1, 2, 3])])
def test_list_with_input_array(func, ms_func, a):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: numpy array'''
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func()
    match_array(res, ms_res, error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('func', [fallback_list_with_input_numpy_tensor])
@pytest.mark.parametrize('ms_func', [ms_fallback_list_with_input_numpy_tensor])
@pytest.mark.parametrize('a', [Tensor([1, 2])])
@pytest.mark.parametrize('b', [[Tensor([1, 2]), Tensor([2, 3])]])
def test_list_with_input_tensor(func, ms_func, a, b):
    """
    Feature: ALL TO ALL
    Description: test cases for builtin list function support in PYNATIVE mode
    Expectation: the result match
    1. Test list() in PYNATIVE mode
    2. give the input data: tensor and (); output tuple
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_func(a, b)
    match_array(res[0], ms_res[0], error=0, err_msg=str(ms_res))
    match_array(res[1], ms_res[1], error=0, err_msg=str(ms_res))
    match_array(res[2], ms_res[2], error=0, err_msg=str(ms_res))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_executed_by_graph():
    """
    Feature: PIJit.
    Description: Support python built-in function list in pijit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def func(x):
        return list((x, x + 1, x + 2))

    x = Tensor([1, 2, 3, 4])
    out = func(x)
    assert isinstance(out, list)
    assert np.all(out[0].asnumpy() == x.asnumpy())
    assert np.all(out[1].asnumpy() == (x + 1).asnumpy())
    assert np.all(out[2].asnumpy() == (x + 2).asnumpy())
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_iteration_applies_relu():
    """
    Feature: Iterate Python list attribute within Cell under JIT.
    Description: Apply ops.relu multiple times by iterating over a stored Python list, then compare pynative and JIT executions.
    Expectation: JIT forward result and gradient match pynative execution.
    Migrated from: test_pijit_list.py::test_pijit_list_with_for_001
    """
    class Net(Cell):
        def __init__(self, lista):
            super().__init__()
            self.lista = lista

        def construct(self, x):
            for _ in self.lista:
                x = ops.relu(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    pynative_input = Tensor(input_np.copy())
    jit_input = Tensor(input_np.copy())

    pynative_net = Net([1, 2])
    pynative_net.set_grad()
    pynative_result = pynative_net(pynative_input)
    grad_net = GradOfFirstInput(pynative_net, sens_param=True)
    grad_net.set_train()
    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = grad_net(pynative_input, sens)

    jit_net = Net([1, 2])
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(jit_input)
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(jit_input, sens)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_conditionals_using_index():
    """
    Feature: Control flow with list indexing inside Cell under JIT.
    Description: Guard different operations with boolean list elements stored on Cell attribute and compare pynative with JIT executions.
    Expectation: JIT forward result and gradient match pynative execution.
    Migrated from: test_pijit_list.py::test_pijit_list_using_index_002
    """
    class Net(Cell):
        def __init__(self, lista):
            super().__init__()
            self.lista = lista

        def construct(self, x):
            if self.lista[0]:
                x = ops.relu(x)
            if self.lista[1]:
                x = ops.square(x)
            return x

    input_np = np.random.randn(2, 3, 4, 5).astype(np.float32)
    pynative_input = Tensor(input_np.copy())
    jit_input = Tensor(input_np.copy())

    pynative_net = Net([True, False])
    pynative_net.set_grad()
    pynative_result = pynative_net(pynative_input)
    grad_net = GradOfFirstInput(pynative_net, sens_param=True)
    grad_net.set_train()
    sens = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = grad_net(pynative_input, sens)

    jit_net = Net([True, False])
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_result = jit_net(jit_input)
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(jit_input, sens)

    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_item_assignment():
    """
    Feature: Python list item assignment within Cell under JIT.
    Description: Modify a list element using an index stored on the Cell and compare the mutated list from pynative and JIT executions.
    Expectation: JIT mutated list matches pynative result.
    Migrated from: test_pijit_list.py::test_pijit_list_mul_index_001
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.idx = 2

        def construct(self, list_x):
            list_x[self.idx] = 9
            return list_x

    input_list = [4, 5, 6]

    pynative_net = Net()
    pynative_input = input_list.copy()
    pynative_result = pynative_net(pynative_input)

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_input = input_list.copy()
    jit_result = jit_net(jit_input)

    expected = [4, 5, 9]
    assert isinstance(pynative_result, list)
    assert isinstance(jit_result, list)
    assert pynative_result == expected
    assert jit_result == expected
    assert jit_result == pynative_result


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_list_slice_assignment():
    """
    Feature: Python list slice assignment within Cell under JIT.
    Description: Assign a Tensor to a slice of a Python list and compare pynative and JIT execution results.
    Expectation: JIT mutated list matches pynative result.
    Migrated from: test_pijit_list.py::test_pijit_list_slice_assign
    """
    class Net(Cell):
        def __init__(self, start=None, end=None, step=None):
            super().__init__()
            self.start = start
            self.end = end
            self.step = step

        def construct(self, a, x):
            a[self.start:self.end:self.step] = x
            return a

    a = [1, 2, 3, 4, 5]
    x = Tensor([11])

    pynative_net = Net(start=1, end=3, step=None)
    pynative_input = a.copy()
    pynative_result = pynative_net(pynative_input, x)

    jit_net = Net(start=1, end=3, step=None)
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_input = a.copy()
    jit_result = jit_net(jit_input, x)

    assert len(pynative_result) == 4
    assert len(jit_result) == 4
    assert pynative_result[0] == 1
    assert jit_result[0] == pynative_result[0]
    assert isinstance(pynative_result[1], Tensor)
    assert isinstance(jit_result[1], Tensor)
    match_array(pynative_result[1], x)
    match_array(jit_result[1], x)
    assert pynative_result[2:] == [4, 5]
    assert jit_result[2:] == pynative_result[2:]
