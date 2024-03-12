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

from mindspore import context, Tensor, jit, ops, mutable
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter
context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path='./log/')

def test_single_if():
    """
    Feature: Contrtol flow inline.
    Description: Inline switch node into kernel graph.
    Expectation: Not throw exception.
    """
    param_a = Parameter(Tensor(5, mstype.int32), name='a')
    param_b = Parameter(Tensor(4, mstype.int32), name='b')

    @jit
    def foo(x, y, param_a, param_b):
        if param_a > param_b:
            param_b += 1
        return x + param_b, y + param_b

    x = Tensor(2, mstype.int32)
    ret1 = foo(x, x, param_a, param_b)
    ret2 = foo(x, x, param_a, param_b)
    assert ret1 == (Tensor(7, mstype.int32), Tensor(7, mstype.int32))
    assert ret2


def test_return_parameter():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """
    param_a = Parameter(Tensor(5))
    param_b = Parameter(Tensor(5))

    @jit
    def foo(x, param_a, param_b):
        if x < 3:
            return param_a
        return param_b

    ret1 = foo(Tensor(1), param_a, param_b)
    assert ret1


def test_return_valuenode():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x):
        if x < 3:
            return 1
        return 2

    ret1 = foo(Tensor(1))
    assert ret1


def test_return_input():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y, z):
        if x < 3:
            return y
        return z

    ret1 = foo(Tensor(1), Tensor(2), Tensor(3))
    assert ret1


def test_value_node_output_in_single_branch():
    """
    Feature: Contrtol flow inline.
    Description: Inline switch node into kernel graph.
    Expectation: Not throw exception.
    """

    @jit
    def BranchReturnTensor(x, y):
        x = x + Tensor(2, mstype.int32)
        y = x + y
        if x < 5:
            return y, Tensor(2, mstype.int32)
        return x, y

    x = Tensor(2, mstype.int32)
    ret1 = BranchReturnTensor(x, x)
    ret2 = BranchReturnTensor(x, x)
    ret3 = BranchReturnTensor(x, x)
    assert ret1
    assert ret2
    assert ret3


def test_diff_ref_count_in_branch():
    """
    Feature: Contrtol flow inline.
    Description: Inline switch node into kernel graph.
    Expectation: Not throw exception.
    """

    @jit
    def BranchDiffRefCount(x, y):
        x = x + Tensor(2, mstype.int32)
        y = x + y
        if x < 5:
            x = x + 3
            y = x + y
        else:
            x = x + 3
            x = x + 4
            x = x + 5
            y = x + y
            y = x + y
            y = x + y
        return x, y

    x = Tensor(2, mstype.int32)
    ret1 = BranchDiffRefCount(x, x)
    x = Tensor(4, mstype.int32)
    ret2 = BranchDiffRefCount(x, x)
    assert ret1
    assert ret2


def test_branch_kernel_backoff():
    """
    Feature: Contrtol flow inline.
    Description: Inline switch node into kernel graph.
    Expectation: Not throw exception.
    """

    @jit
    def foo(x, y, shape):
        x = x + Tensor(2, mstype.int32)
        if y < 5:
            z = ops.reshape(x, shape)
        else:
            z = x
        return z + 1

    x = Tensor([2, 2, 2, 2, 2, 2], mstype.int32)
    y = Tensor(2, mstype.int32)
    ret1 = foo(x, y, mutable((2, 3)))
    ret2 = foo(x, y, mutable((2, 3)))
    ret3 = foo(x, y, mutable((2, 3)))
    assert ret1[0][0]
    assert ret2[0][0]
    assert ret3[0][0]


def test_update_parameter():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))

    @jit
    def foo(x, param_a):
        x = x + param_a
        if x < 3:
            param_a = param_a + 2
        else:
            param_a = param_a + x
        return param_a

    ret1 = foo(Tensor(1), param_a)
    ret2 = foo(Tensor(1), param_a)
    ret3 = foo(Tensor(1), param_a)
    assert ret1
    assert ret2
    assert ret3


def test_update_and_return_parameter():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))
    param_b = Parameter(Tensor(5))

    @jit
    def foo(x, param_a, param_b):
        x = x + param_a
        if x < 3:
            param_a = param_a + 2
            param_b = param_b - param_a
            return Tensor(2), param_b
        param_a = param_a + x
        param_b = param_b + param_a
        return param_a, param_b

    ret1 = foo(Tensor(1), param_a, param_b)
    ret2 = foo(Tensor(1), param_a, param_b)
    ret3 = foo(Tensor(1), param_a, param_b)
    assert ret1
    assert ret2
    assert ret3


def test_return_switch_input_in_branch():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))
    param_b = Parameter(Tensor(5))

    @jit
    def foo(x, param_a, param_b):
        x = x + param_a
        if x < 3:
            param_a = param_a + 2
            param_b = param_b - param_a
            return x, param_b
        param_a = param_a + x
        param_b = param_b + param_a
        return param_a, param_b

    ret1 = foo(Tensor(1), param_a, param_b)
    ret2 = foo(Tensor(1), param_a, param_b)
    ret3 = foo(Tensor(1), param_a, param_b)
    assert ret1
    assert ret2
    assert ret3


def test_return_switch_input():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))
    param_b = Parameter(Tensor(5))

    @jit
    def foo(x, param_a, param_b):
        x = x + param_a
        if x < 3:
            param_a = param_a + 2
            param_b = param_b - param_a
        else:
            param_a = param_a + x
            param_b = param_b + param_a
        return x, param_b, 3

    ret1 = foo(Tensor(1), param_a, param_b)
    ret2 = foo(Tensor(1), param_a, param_b)
    ret3 = foo(Tensor(1), param_a, param_b)
    assert ret1
    assert ret2
    assert ret3


def test_tuple_args_to_dynamic_tuple_para():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y):
        y_shape = ops.shape(y)
        if x < 3:
            y_shape = y_shape * 2
        else:
            y_shape = y_shape * 3
        return y_shape[0]

    ret1 = foo(Tensor(1), Tensor([[6, 6, 6], [6, 6, 6]]))
    ret2 = foo(Tensor(1), Tensor([[6, 6, 6], [6, 6, 6]]))
    ret3 = foo(Tensor(1), Tensor([[6, 6, 6], [6, 6, 6]]))
    assert ret1
    assert ret2
    assert ret3


def test_tuple_input_to_switch():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y, dst_shape):
        y, _ = ops.unique(y)
        y = ops.reshape(y, dst_shape)
        y_shape = ops.shape(y)
        if x < 3:
            y_shape = y_shape * 2
        else:
            y_shape = y_shape * 3
        return y_shape

    ret1 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36]]), mutable((2, 3)))
    ret2 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36]]), mutable((2, 3)))
    ret3 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36]]), mutable((2, 3)))
    assert ret1[0]
    assert ret2[0]
    assert ret3[0]


def test_dynamic_tuple_input_to_switch():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, dyn_tuple):
        if x < 3:
            dyn_tuple = dyn_tuple * 2
        else:
            dyn_tuple = dyn_tuple * 3
        return dyn_tuple

    ret1 = foo(Tensor(1), mutable((2, 3), dynamic_len=True))
    ret2 = foo(Tensor(1), mutable((2, 3), dynamic_len=True))
    ret3 = foo(Tensor(1), mutable((2, 3), dynamic_len=True))
    assert ret1
    assert ret2
    assert ret3


def test_return_condition():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, cond):
        if cond:
            x = x * 2
            return x, cond
        x = x * 3
        return x, cond

    ret1 = foo(Tensor(1), Tensor(True))
    ret2 = foo(Tensor(1), Tensor(True))
    ret3 = foo(Tensor(1), Tensor(True))
    assert ret1
    assert ret2
    assert ret3


def test_return_include_other_output():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y):
        y = y + 2
        y = y * 3
        y = y / 4
        y = y - 5
        y = y * y
        if x < 5:
            x = x * 2
        else:
            x = x + 2
        return x, y

    ret1 = foo(Tensor(1), Tensor(2))
    ret2 = foo(Tensor(1), Tensor(2))
    ret3 = foo(Tensor(1), Tensor(2))
    assert ret1
    assert ret2
    assert ret3


def test_branch_output_include_refnode():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y, dst_shape):
        y, _ = ops.unique(y)
        y = ops.reshape(y, dst_shape)
        if x < 3:
            y = ops.expand_dims(y, 1)
            y = ops.flatten(y)
        return y

    ret1 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36], [6, 18, 36]]), mutable((2, 3)))
    ret2 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36]]), mutable((2, 3)))
    ret3 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36]]), mutable((2, 3)))
    assert ret1[0][0]
    assert ret2[0][0]
    assert ret3[0][0]


def test_include_dynamic_shape():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y):
        y, _ = ops.unique(y)
        if x < 3:
            y = y * 2
        else:
            z1 = y / 6
            z2 = y * 2
            z3 = y - Tensor([[6, 12, 18], [24, 30, 36]])
            z4 = y + Tensor([[1, 2, 3], [4, 5, 6]])
            y = z1 + z2 + z3 + z4
        return y

    ret1 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36], [6, 18, 36]]))
    ret2 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36], [12, 18, 30], [18, 24, 36]]))
    ret3 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36]]))
    print("*********************************", ret1)
    assert ret1[0]
    assert ret2[0]
    assert ret3[0]


def test_control_arrow_from_switch_to_gather():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """
    param_a = Parameter(Tensor(5))
    param_b = Parameter(Tensor(5))

    @jit
    def foo(x, param_a, param_b):
        x = x + param_a
        if x < 3:
            param_a = param_a + 2
            param_b = param_b - param_a
            return Tensor(2), param_b
        x = x + param_a
        return param_a, param_b

    ret1 = foo(Tensor(1), param_a, param_b)
    ret2 = foo(Tensor(1), param_a, param_b)
    ret3 = foo(Tensor(1), param_a, param_b)
    assert ret1
    assert ret2
    assert ret3


def test_branch_only_u_input():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y):
        x = x + 1
        if x < 3:
            ops.print("this is true")
        else:
            y = ops.reshape(y, (4, 1))
            ops.print("this is false")
        return ops.shape(y)

    ret1 = foo(Tensor(1), Tensor([[1, 2], [3, 4]]))
    assert ret1


def test_branch_u_input_and_input():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y):
        x = x + 1
        if x < 3:
            ops.print("this is true")
        else:
            y = ops.reshape(y, (4, 1))
            ops.print("this is false")
        return ops.shape(y)

    ret1 = foo(Tensor(1), Tensor([[1, 2], [3, 4]]))
    assert ret1


def test_branch_output_real_tuple():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.

    Expectation: AttributeError.
    """

    @jit
    def foo(x, y):
        if x < 3:
            y, _ = ops.unique(y)
            y = ops.expand_dims(y, 1)
            y = ops.flatten(y)
            z = ops.shape(y)
        else:
            z = ops.shape(y)
        return z

    ret1 = foo(Tensor(1), Tensor([[6, 12, 18], [24, 30, 36], [6, 18, 36]]))
    ret2 = foo(Tensor(5), Tensor([[6, 12, 18], [24, 30, 36]]))
    assert ret1
    assert ret2


def test_branch_output_dynamic_tuple():
    """
    Feature: Contrtol flow inline.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    @jit
    def foo(x, y, shape):
        if y < 5:
            z = ops.reshape(x, shape)
            out = ops.shape(z)
        else:
            out = ops.shape(x)
        return out

    x = Tensor([2, 2, 2, 2, 2, 2], mstype.int32)
    y = Tensor(2, mstype.int32)
    ret1 = foo(x, y, mutable((2, 3), dynamic_len=True))
    assert ret1[0]
