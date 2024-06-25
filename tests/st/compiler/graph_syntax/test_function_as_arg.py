# Copyright 2023 Huawei Technologies Co., Ltd
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
""" test graph function as input argument. """
import pytest
import numpy as np

from mindspore import Tensor, jit, ops
import mindspore as ms
from tests.mark_utils import arg_mark


class AnyCallable:
    def __init__(self, x):
        self.x = x

    def __call__(self, x):
        return ops.dtype(x)


class MyClass:
    @jit
    def forward2(self, *args, **kw):
        return self(*args, **kw)

    def __call__(self, *args, **kw):
        return args[0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_call_interpret_obj_in_jit_function():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: The PyExecute seemingly be eliminated, and last call a ClassType eventually cause this error.
    """
    @jit
    def func(x):
        a = AnyCallable(Tensor([1, 2, 4]))
        return a(x)

    a = func(Tensor([1, 2]))
    assert a == ms.int64


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_callable_in_unpack_args():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: No exception.
    """
    func = MyClass.forward2  # jit function

    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    call = AnyCallable(Tensor([1, 2]))
    a = forward1(call, Tensor([1, 2]))
    assert a == ms.int64


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_callable_obj():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: No exception.
    """

    func = MyClass.forward2

    @jit
    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    call = AnyCallable(Tensor([1, 2]))
    a = forward1(call, Tensor([1, 2]))
    assert a == ms.int64


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_callable_obj_1():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: No exception.
    """

    func = MyClass.forward2  # jit_function

    @jit
    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    a = forward1(MyClass(), Tensor([1, 2]))
    assert np.allclose(a.asnumpy(), np.array([1, 2]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_callable_obj_2():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: the MyClass has jit decorator resolved as a function using NameSpace.
    """

    func = MyClass.forward2  # jit_function

    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    with pytest.raises(ValueError) as raise_info:
        a = forward1(MyClass(), Tensor([1, 2]))
        assert np.allclose(a.asnumpy(), np.array([1, 2]))
    assert "The object is not callable. Please check code" in str(raise_info.value)



@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_primitive():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: the MyClass has jit decorator resolved as a function using NameSpace.
    """

    func = MyClass.forward2  # jit_function

    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    a = forward1(ops.operations.ReLU(), Tensor([1, 2]))
    assert np.allclose(a.asnumpy(), np.array([1, 2]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_cannot_compile_func():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: the MyClass has jit decorator resolved as a function using NameSpace.
    """

    func = MyClass.forward2  # jit_function

    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    import copy
    a = forward1(copy.deepcopy, Tensor([1, 2]))
    assert np.allclose(a.asnumpy(), np.array([1, 2]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_func_with_fallback():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: the MyClass has jit decorator resolved as a function using NameSpace.
    """

    func = MyClass.forward2  # jit_function

    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    import copy

    def func1(x):
        return copy.deepcopy(x) + x

    a = forward1(func1, Tensor([1, 2]))
    assert np.allclose(a.asnumpy(), np.array([2, 4]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_function_input_a_user_def_func():
    """
    Feature: Support input args as function.
    Description: Support input args as function.
    Expectation: the MyClass has jit decorator resolved as a function using NameSpace.
    """

    func = MyClass.forward2  # jit_function

    def forward1(*args, **kwargs):
        return func(*args, **kwargs)

    def ffff(*args):
        return args[0]

    a = forward1(ffff, Tensor([1, 2]))
    assert np.allclose(a.asnumpy(), np.array([1, 2]))
