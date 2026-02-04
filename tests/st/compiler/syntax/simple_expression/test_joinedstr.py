# Copyright 2025 Huawei Technologies Co., Ltd
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
""" test syntax raise and JoinedStr ops in strict mode"""
import os
import pytest
import numpy as np

from mindspore import ops
from mindspore import Tensor
from mindspore import context
from mindspore import mutable
from mindspore.graph.api import jit
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_1():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        raise ValueError(f"The input can not be {x}.")
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be [1]." in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_2():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        raise ValueError(f"The input can not be {x}", ".")
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "('The input can not be [1]', '.')" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_constant_tuple():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = (Tensor([1]), Tensor([2]), Tensor([3]))
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be (Tensor(shape=[1]," in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_constant_list():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = [Tensor([1]), Tensor([2]), Tensor([3])]
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be [Tensor(shape=[1]," in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_constant_dict():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = {"a": Tensor([1]), "b": Tensor([2])}
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be {'a': " in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_tuple():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = (x, 1)
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be ([1], 1)" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_nested_tuple():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = ((x, 0), 1)
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be (([1], 0), 1)" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_list():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = [x, 1]
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be [[1], 1]" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_nested_list():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = [[x, 0], 1]
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be [[[1], 0], 1]" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_nested_sequence():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = ([x, 0], 1)
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x)
        print(res)
    assert "The input can not be ([[1], 0], 1)" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_dict():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = {"1": x}
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(RuntimeError) as raise_info:
        res = func(input_x)
        print(res)
    assert "For JoinedStr, do not support dict input with variable elements" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_variable_nested_dict():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x):
        y = (1, {"1": x})
        if x < Tensor([2]):
            raise ValueError(f"The input can not be {y}.")
        return x
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    with pytest.raises(RuntimeError) as raise_info:
        res = func(input_x)
        print(res)
    assert "For JoinedStr, do not support dict input with variable elements" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([2])
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x, input_y)
        print(res)
    assert "The input can not be [1]." in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_joinedstr_control_flow():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return x + y
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([0])
    res = func(input_x, input_y)
    assert np.all(res.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_control_flow_grad_in_graph():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return x + y

    @jit
    def grad_func(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([2])
    with pytest.raises(ValueError) as raise_info:
        res = grad_func(func, input_x, input_y)
        print(res)
    assert "The input can not be [1]." in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_control_flow_grad_in_pynative():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func_raise_error(x, y):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return x + y

    def grad_func_raise_error(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([2])
    with pytest.raises(ValueError) as raise_info:
        res = grad_func_raise_error(func_raise_error, input_x, input_y)
        print(res)
    assert "The input can not be [1]." in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_joinedstr_control_flow_grad_in_graph():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return x + y

    @jit
    def grad_func(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([0])
    res = grad_func(func, input_x, input_y)
    assert np.all(res.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_joinedstr_control_flow_grad_in_pynative():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func_not_raise_error(x, y):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return x + y

    def grad_func_not_raise_error(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([0])
    res = grad_func_not_raise_error(func_not_raise_error, input_x, input_y)
    assert np.all(res.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_scalar_join_grad_in_graph():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y, z):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return z

    @jit
    def grad_func(foo, x, y, z):
        return ops.grad(foo)(x, y, z)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([2])
    input_z = 1
    with pytest.raises(ValueError) as raise_info:
        res = grad_func(func, input_x, input_y, input_z)
        print(res)
    assert "The input can not be [1]." in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_joinedstr_scalar_join_grad_in_graph():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y, z):
        if x < y:
            raise ValueError(f"The input can not be {x}.")
        return z

    @jit
    def grad_func(foo, x, y, z):
        return ops.grad(foo)(x, y, z)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([0])
    input_z = 1
    res = grad_func(func, input_x, input_y, input_z)
    assert np.all(res.asnumpy() == np.array([0]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_multi_input_control_flow_grad_in_pynative():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if x < y:
            raise ValueError(f"The input is {x} ", "and", f"{y}")
        return x + y

    def grad_func(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([1])
    input_y = Tensor([2])
    with pytest.raises(ValueError) as raise_info:
        res = grad_func(func, input_x, input_y)
        print(res)
    assert "('The input is [1] ', 'and', '[2]')" in str(raise_info.value)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_not_raise_joinedstr_multi_input_control_flow_grad_in_pynative():
    """
    Feature: Test raise syntax in strict mode.
    Description: Test raise syntax in strict mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if x < y:
            raise ValueError(f"The input is {x} ", "and", f"{y}")
        return x + y

    def grad_func(foo, x, y):
        return ops.grad(foo)(x, y)

    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '0'
    input_x = Tensor([2])
    input_y = Tensor([1])
    res = grad_func(func, input_x, input_y)
    assert np.all(res.asnumpy() == np.array([1]))
    os.environ['MS_DEV_JIT_SYNTAX_LEVEL'] = '2'


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_joinedstr_infer_type():
    """
    Feature: Test joinedstr infer type.
    Description: Test joinedstr infer type.
    Expectation: Success.
    """
    @jit
    def func(x):
        x = ops.add(x, x)
        format_f = f'format_f ={x}'
        assert isinstance(format_f, str)
        return x

    input_x = Tensor([1])
    res = func(input_x)
    assert res == Tensor([2])


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_mutable_list_tensor():
    """
    Feature: Test raise syntax in lax mode.
    Description: Test raise syntax in lax mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if y >= 1:
            raise ValueError(f"The input can not be {x}.")
        return y + 1

    input_x = mutable([Tensor([1]), Tensor([2]), Tensor([3])])
    input_y = Tensor(1)
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x, input_y)
        print(res)
    assert "The input can not be [[1], [2], [3]]." in str(raise_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_raise_joinedstr_mutable_nested_sequence_tensor():
    """
    Feature: Test raise syntax in lax mode.
    Description: Test raise syntax in lax mode.
    Expectation: Throw correct exception when needed.
    """
    @jit
    def func(x, y):
        if y >= 1:
            raise ValueError(f"The input can not be {x}.")
        return y + 1

    input_x = mutable(([Tensor([1]), Tensor([2])], Tensor([3])))
    input_y = Tensor(1)
    with pytest.raises(ValueError) as raise_info:
        res = func(input_x, input_y)
        print(res)
    assert "The input can not be ([[1], [2]], [3])." in str(raise_info.value)
