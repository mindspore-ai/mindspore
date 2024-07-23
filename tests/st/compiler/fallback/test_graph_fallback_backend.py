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
import pytest
from mindspore import Tensor, context, jit, ops, Parameter
from mindspore.ops.operations._sequence_ops import SequenceAddN
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)

class ListClass:
    def __init__(self):
        self.x = [Tensor(1), Tensor(2), Tensor(3), Tensor(4)]

list_obj = ListClass()

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_base():
    """
    Feature: Backend support any type.
    Description: Base scene.
    Expectation: AttributeError.
    """

    @jit
    def foo():
        a = list_obj.x
        return ops.addn(a)

    ret = foo()
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_add_weight():
    """
    Feature: Backend support any type.
    Description: Add weight.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))

    @jit
    def foo():
        a = list_obj.x
        b = ops.addn(a)
        c = b + param_a
        d = SequenceAddN()(a)
        e = d - param_a
        return param_a + c + e

    ret1 = foo()
    ret2 = foo()
    assert ret1
    assert ret2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_call_twice():
    """
    Feature: Backend support any type.
    Description: Call single function twice.
    Expectation: AttributeError.
    """

    @jit
    def foo():
        a = list_obj.x
        return ops.addn(a)

    ret1 = foo()
    ret2 = foo()
    assert ret1
    assert ret2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_return_tuple():
    """
    Feature: Backend support any type.
    Description: Return tuple value.
    Expectation: AttributeError.
    """

    @jit
    def foo():
        a = list_obj.x
        b = ops.addn(a)
        d = SequenceAddN()(a)
        return (b, d)

    ret1 = foo()
    ret2 = foo()
    assert ret1
    assert ret2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_if():
    """
    Feature: Backend support any type.
    Description: Control flow if.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))

    @jit
    def foo(x):
        a = list_obj.x
        b = ops.addn(a)
        if x > 3:
            c = b + param_a + x
            d = SequenceAddN()(a)
            e = d - param_a + c
        else:
            c = b - param_a + x
            d = SequenceAddN()(a)
            e = d + x + param_a
        return param_a + c + e

    ret1 = foo(Tensor(1))
    ret2 = foo(Tensor(5))
    assert ret1
    assert ret2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_while():
    """
    Feature: Backend support any type.
    Description: Controlflow while.
    Expectation: AttributeError.
    """

    param_a = Parameter(Tensor(5))

    @jit
    def foo(x):
        a = list_obj.x
        b = ops.addn(a)
        e = x
        while x < 3:
            b = b + param_a + x
            d = SequenceAddN()(a)
            e = e - param_a + d
            x = x + 1
        return param_a + b + e

    ret1 = foo(Tensor(1))
    ret2 = foo(Tensor(0))
    assert ret1
    assert ret2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_input_value_node():
    """
    Feature: Backend support any type.
    Description: Value node input for pyexecute.
    Expectation: AttributeError.
    """

    @jit
    def foo():
        a = list_obj.x
        b = ops.addn(a)
        c = a[0]
        return b + c

    ret = foo()
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_heter():
    """
    Feature: Backend support any type.
    Description: Heter in function.
    Expectation: AttributeError.
    """

    context.set_context(device_target="GPU")
    mul_op = ops.Mul().add_prim_attr("primitive_target", "CPU")

    @jit
    def foo():
        a = list_obj.x
        b = ops.addn(a)
        d = SequenceAddN()(a)
        return mul_op(b, d)

    ret = foo()
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_output_batch():
    """
    Feature: Backend support any type.
    Description: Batch output arrow in actor.
    Expectation: AttributeError.
    """

    @jit
    def foo():
        a = list_obj.x
        b = ops.addn(a)
        d = SequenceAddN()(a)
        return (b, d, d)

    ret = foo()
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_output_parameter():
    """
    Feature: Backend support any type.
    Description: Parameter in tuple output.
    Expectation: AttributeError.
    """

    @jit
    def foo(x):
        a = list_obj.x
        b = ops.addn(a)
        d = SequenceAddN()(a)
        return (b, d, x)

    ret = foo(Tensor(1))
    assert ret


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fallback_output_valuenode():
    """
    Feature: Backend support any type.
    Description: Value node in tuple output.
    Expectation: AttributeError.
    """

    @jit
    def foo():
        a = list_obj.x
        b = ops.addn(a)
        d = SequenceAddN()(a)
        return (b, d, 1)

    ret = foo()
    assert ret
