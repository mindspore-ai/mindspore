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
"""Test grad jit with filter level 2"""
from mindspore import context
from mindspore import ops
from mindspore import jit
from mindspore import Tensor
from mindspore.graph._parse import compile_config
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_psjit():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret1 == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_psjit_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_psjit_with_multiple_position():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_psjit_with_multiple_position_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_pijit():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret1 == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_pijit_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret = ops.grad(foo)(x, y, z)  # pylint: disable=not-callable
    assert ret == 1
    ret2 = ops.grad(foo)(x + 1, y, z)  # pylint: disable=not-callable
    assert ret2 == 1
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_pijit_with_multiple_position():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return a * 1 + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_level2_filter_grad_jit_pijit_with_multiple_position_2():
    """
    Feature: Test filter grad jit graph.
    Description: Test filter grad jit graph in psjit.
    Expectation: No exception.
    """
    @jit(capture_mode="bytecode")
    def inner(a, b, c, d):
        return ops.relu(a) + b * 2 + c * 3 + d * 4

    def foo(x, y, z):
        x = x + 1
        return inner(x, y, z, z)

    compile_config.GRAD_JIT_FILTER = "2"
    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1])
    y = Tensor([2])
    z = Tensor([3])
    ret1 = ops.grad(foo, grad_position=(0, 1))(x, y, z)  # pylint: disable=not-callable
    assert isinstance(ret1, tuple)
    assert len(ret1) == 2
    assert ret1[0] == 1
    assert ret1[1] == 2
    ret2 = ops.grad(foo, grad_position=(0, 1))(x + 1, y, z)  # pylint: disable=not-callable
    assert isinstance(ret2, tuple)
    assert len(ret2) == 2
    assert ret2[0] == 1
    assert ret2[1] == 2
    compile_config.GRAD_JIT_FILTER = ""
