# Copyright 2022-2024 Huawei Technologies Co., Ltd
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
"""test verification and repeat compile of mutable input"""
import numpy as np
import pytest
from mindspore.common import mutable
from mindspore.common.api import _CellGraphExecutor, _MindsporeFunctionExecutor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore._c_expression import Tensor as Tensor_
from mindspore import Parameter
from mindspore import jit


def seq_compare(a, b):
    if isinstance(a, (list, tuple)):
        if not a and b:
            return False
        for aa, bb in zip(a, b):
            if not seq_compare(aa, bb):
                return False
        return True
    if isinstance(a, Tensor):
        return np.allclose(a.asnumpy(), b.asnumpy())
    if isinstance(a, float):
        return np.allclose(a, b)
    return a == b


def dict_compare(a, b):
    if isinstance(a, dict):
        if not a and b:
            return False
        for aa, bb in zip(a.values(), b.values()):
            if not seq_compare(aa, bb):
                return False
        return True
    if isinstance(a, Tensor):
        return np.allclose(a.asnumpy(), b.asnumpy())
    if isinstance(a, float):
        return np.allclose(a, b)
    return a == b


def generate_argument_with_mutable(args):
    new_args = []
    for x in args:
        new_args.append(mutable(x))
    return new_args


def compare_compile_phase(net, args1, args2, const_arg=True):
    _cell_graph_executor = _CellGraphExecutor()
    phase1, _ = _cell_graph_executor.compile(net, *args1)
    phase2, _ = _cell_graph_executor.compile(net, *args2)
    if const_arg:
        assert phase1 != phase2
    else:
        assert phase1 == phase2

    fn = net.construct
    _mindspore_function_executor = _MindsporeFunctionExecutor(fn, 0)
    fn_name = fn.__name__
    phase1 = _mindspore_function_executor.compile(fn_name, *args1)
    phase2 = _mindspore_function_executor.compile(fn_name, *args2)
    if const_arg:
        assert phase1 != phase2
    else:
        assert phase1 == phase2

    new_args1 = generate_argument_with_mutable(args1)
    new_args2 = generate_argument_with_mutable(args2)
    phase1, _ = _cell_graph_executor.compile(net, *new_args1)
    phase2, _ = _cell_graph_executor.compile(net, *new_args2)
    assert phase1 == phase2

    phase1 = _mindspore_function_executor.compile(fn_name, *new_args1)
    phase2 = _mindspore_function_executor.compile(fn_name, *new_args2)
    assert phase1 == phase2


def test_mutable_wrong_input_in_graph():
    """
    Feature: create and return mutable object in graph mode.
    Description: test mutable input in graph.
    Expectation: raise exception.
    """

    @jit
    def foo1():
        return mutable()

    @jit
    def foo2():
        return mutable([1], [1])

    @jit
    def foo3():
        return mutable(True, True)

    @jit
    def foo4():
        return mutable([1, np.array([[1, 2, 3]])])

    @jit
    def foo5():
        x = mutable([1, 2, 3], True)
        return mutable(x, False)

    @jit
    def foo6():
        return mutable([1, 2.2, 3], True)


    with pytest.raises(RuntimeError) as ex1:
        foo1()
    assert "For 'mutable', the number of inputs should be 1 or 2, but got" in str(ex1.value)
    with pytest.raises(TypeError) as ex2:
        foo2()
    assert "For 'mutable', the second input should be bool, but got" in str(ex2.value)
    with pytest.raises(TypeError) as ex3:
        foo3()
    assert "For 'mutable', when the variable_len is True, the first input should be "\
           "list or tuple, but got" in str(ex3.value)
    with pytest.raises(TypeError) as ex4:
        foo4()
    assert "For 'mutable', the 'input_data' should be one of (bool, int, float, Tensor, "\
           "tuple, list, dict) or their nested structures, but got" in str(ex4.value)
    with pytest.raises(RuntimeError) as ex5:
        foo5()
    assert "For 'mutable', can not convert a dynamic length sequence to constant length."\
        in str(ex5.value)
    with pytest.raises(ValueError) as ex6:
        foo6()
    assert "In graph mode, the element type of dynamic length array must be the same."\
        in str(ex6.value)


def test_mutable_wrong_input_out_of_graph():
    """
    Feature: create and return mutable object.
    Description: test mutable input out of graph.
    Expectation: raise exception.
    """
    @jit
    def foo1(x):
        return x

    with pytest.raises(TypeError) as ex1:
        foo1(mutable([1], [1]))
    assert "For 'mutable', the second input should be bool, but got" in str(ex1.value)
    with pytest.raises(TypeError) as ex2:
        foo1(mutable(True, True))
    assert "For 'mutable', when the variable_len is True, the first input should be "\
           "list or tuple, but got" in str(ex2.value)
    with pytest.raises(TypeError) as ex3:
        foo1(mutable([1, np.array([[1, 2, 3]])]))
    assert "For 'mutable', the 'input_data' should be one of (bool, int, float, Tensor, "\
           "tuple, list, dict) or their nested structures" in str(ex3.value)
    with pytest.raises(ValueError) as ex4:
        foo1(mutable([1, 2.2, 3], True))
    assert "In graph mode, the element type of dynamic length array must be the same."\
        in str(ex4.value)


def test_mutable_wrong_input_self_reference():
    """
    Feature: create and return mutable object.
    Description: test mutable input out of graph.
    Expectation: raise exception.
    """
    input_list = []
    input_list.append(input_list)
    with pytest.raises(TypeError) as ex:
        mutable(input_list)
    assert "with no self-reference" in str(ex.value)


def test_mutable_input_with_bool():
    """
    Feature: Set Constants mutable.
    Description: Set mutable for bool value.
    Expectation: No Exception.
    """
    mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32), (True,)])

    data = mutable(True)
    assert isinstance(data, bool) and data
    assert mutable(True) & False is False
    assert mutable(False) | True is True
    assert mutable(False) ^ False is False
    assert str(mutable(True)) == "True"

    @jit
    def net(data):
        x = mutable(False)
        return x, data, F.isconstant(x), F.isconstant(data)
    out = net(data)
    assert isinstance(out[0], bool) and not out[0]
    # type(out[1]): <Class 'mindspore.common.mutable._Bool'>
    assert isinstance(out[1], int) and out[1]
    assert not out[2]
    assert not out[3]


def test_mutable_input_with_scalar():
    """
    Feature: Set Constants mutable.
    Description: Set mutable for scalar.
    Expectation: No Exception.
    """

    @jit
    def foo(x1, x2):
        y1 = mutable(1)
        y2 = mutable(1.3)
        out = [x1, x2, y1, y2]
        for x in (x1, x2, y1, y2):
            out.append(F.isconstant(x))
        return out

    x1 = mutable(1)
    x2 = mutable(1.3)
    out = foo(x1, x2)
    assert isinstance(out[0], int) and out[0] == x1
    assert isinstance(out[1], float) and np.allclose(out[1], x2)
    assert isinstance(out[2], int) and out[2] == x1
    assert isinstance(out[3], float) and np.allclose(out[3], x2)
    for ele in out[4:]:
        assert not ele


def test_mutable_input_with_sequence():
    """
    Feature: Set Constants mutable.
    Description: Set mutable for list or tuple.
    Expectation: No Exception.
    """
    @jit
    def foo(list1, tuple1):
        list2 = mutable([True, 1, 1.3, Tensor([1, 2, 3])])
        tuple2 = mutable((True, 1, 1.3, Tensor([1, 2, 3])))
        out = [list1, tuple1, list2, tuple2]
        for seq in (list1, tuple1, list2, tuple2):
            for ele in seq:
                out.append(F.isconstant(ele))
        return out

    list1 = mutable([True, 1, 1.3, Tensor([1, 2, 3])])
    tuple1 = mutable((True, 1, 1.3, Tensor([1, 2, 3])))
    out = foo(list1, tuple1)
    assert isinstance(out[0], list) and seq_compare(out[0], list1)
    assert isinstance(out[1], tuple) and seq_compare(out[1], tuple1)
    assert isinstance(out[2], list) and seq_compare(out[2], list1)
    assert isinstance(out[3], tuple) and seq_compare(out[3], tuple1)
    for ele in out[4:]:
        assert not ele


def test_scalar_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for scalar input twice are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, scalr_input):
            out = scalr_input + self.z
            return out

    x = 1
    y = 2
    net = Net()
    compare_compile_phase(net, [x], [y])


def test_sequence_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for tuple[any] and list[any] input twice
                are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, tuple_seq):
            x = tuple_seq[0]
            x = x * tuple_seq[1] * tuple_seq[2][1] * tuple_seq[2][1]
            y = tuple_seq[4]['a']
            x = x * self.z
            out = self.matmul(x, y)
            return out

    input_tensor1 = Tensor([[0.1, 0.2, 0.3], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    input_scalar1 = 1.3
    input_tuple1 = (1, 2, 3)
    input_list1 = [2, 3, 4]
    input_dict1 = {'a': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 0.3], [2.1, 1.2, 3.3]]),
                   'b': input_scalar1,
                   'c': input_tuple1}
    input_tensor2 = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    input_scalar2 = 2.7
    input_tuple2 = (5, 6, 7)
    input_list2 = [7, 8, 9]
    input_dict2 = {'a': Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 0.3], [2.1, 1.2, 3.3]]),
                   'b': input_scalar1,
                   'c': input_tuple1}
    net = Net()

    args1 = [input_tensor1, input_scalar1, input_tuple1, input_list1, input_dict1]
    args2 = [input_tensor2, input_scalar2, input_tuple2, input_list2, input_dict2]
    compare_compile_phase(net, [args1], [args2])

    args1 = (input_tensor1, input_scalar1, input_tuple1, input_list1, input_dict1)
    args2 = (input_tensor2, input_scalar2, input_tuple2, input_list2, input_dict2)
    compare_compile_phase(net, [args1], [args2])


def test_dict_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for dict(any) input twice are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, input_dict):
            x = input_dict['a']
            y = input_dict['b'][0]
            w = input_dict['c']
            x = x * self.z
            x = x * w
            out = self.matmul(x, y)
            return out

    x = Tensor([[0.1, 0.2, 0.3], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = (Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 0.3], [2.1, 1.2, 3.3]], dtype=mstype.float32),)
    w = 1
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = (Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32),)
    o = 2
    net = Net()

    compare_compile_phase(net, [{'a': x, 'b': y, 'c': w}], [{'a': p, 'b': q, 'c': o}])


def test_tensor_inputs_compile_phase():
    """
    Feature: Set Constants mutable.
    Description: Test whether the compilation phase for Tensor input twice are the same.
    Expectation: The phases are the same.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        def construct(self, x, y):
            x = x * self.z
            out = self.matmul(x, y)
            return out

    net = Net()

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    q = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    compare_compile_phase(net, (x, y), (p, q), const_arg=False)

    x.set_const_arg(True)
    y.set_const_arg(True)
    p.set_const_arg(True)
    q.set_const_arg(True)
    compare_compile_phase(net, (x, y), (p, q), const_arg=True)

    x_ = Tensor_(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32))
    y_ = Tensor_(Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))
    p_ = Tensor_(Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32))
    q_ = Tensor_(Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32))
    compare_compile_phase(net, (x_, y_), (p_, q_), const_arg=False)


def test_const_arg_tensor_inputs_compile_phase():
    """
    Feature: Set constant tensor input to mutable.
    Description: Test whether the compilation phase for tensor const_arg inputs twice are the same.
    Expectation: The phases are the same only when the tensor inputs are set mutable.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.matmul = P.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

    net = Net()
    # Init the tensors as const arguments.
    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]],\
        dtype=mstype.float32, const_arg=True)
    p = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32, const_arg=True)
    q = Tensor([[0.01, 3.0, 1.1], [1.0, 0.2, 1.3], [2.1, 1.2, 3.3]],\
        dtype=mstype.float32, const_arg=True)
    compare_compile_phase(net, (x, y), (p, q), const_arg=True)

    x.set_const_arg(False)
    y.set_const_arg(False)
    p.set_const_arg(False)
    q.set_const_arg(False)
    compare_compile_phase(net, (x, y), (p, q), const_arg=False)
