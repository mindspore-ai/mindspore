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
"""Test use pijit and psjit together"""

import numpy as np

import mindspore as ms
from mindspore import Tensor, ops

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import match_array


class _PlainAddNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.scale = 1

    def func(self, x):
        y = x.asnumpy() + x.asnumpy()
        return Tensor(y) * self.scale

    def construct(self, x):
        return self.func(x)


class _BytecodeOuterAstInnerNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.scale = 1

    @ms.jit(capture_mode="ast")
    def func(self, x):
        y = x.asnumpy() + x.asnumpy()
        return Tensor(y) * self.scale

    @ms.jit(capture_mode="bytecode")
    def construct(self, x):
        return self.func(x)


class _AstOuterBytecodeInnerNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.scale = 1

    @ms.jit(capture_mode="bytecode")
    def func(self, x):
        y = x.asnumpy() + x.asnumpy()
        return Tensor(y) * self.scale

    @ms.jit(capture_mode="ast")
    def construct(self, x):
        return self.func(x)


class _PlainTrigNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.scale = 1

    def func(self, x):
        return ops.square(x) * self.scale

    def construct(self, x):
        y = ops.sin(x)
        z = self.func(y)
        return ops.cos(z)


class _BytecodeFuncTrigNet(_PlainTrigNet):
    @ms.jit(capture_mode="bytecode")
    def func(self, x):
        return ops.square(x) * self.scale


class _PlainTwoStageTrigNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.scale = 1

    def func1(self, x):
        return ops.square(x) * self.scale

    def func2(self, x):
        return ops.sin(x) * self.scale

    def construct(self, x):
        y = self.func1(x)
        z = self.func2(y)
        return ops.cos(z)


class _MixedDecoratorTrigNet(_PlainTwoStageTrigNet):
    @ms.jit(capture_mode="bytecode")
    def func1(self, x):
        return ops.square(x) * self.scale

    @ms.jit(capture_mode="ast")
    def func2(self, x):
        return ops.sin(x) * self.scale


class _PlainShapeAwareProductNet(ms.nn.Cell):
    def __init__(self):
        super().__init__()
        self.rank = 2

    def construct(self, x, y):
        if len(x.shape) == self.rank:
            x = x + y
        return x * y


class _PsJitShapeAwareProductNet(_PlainShapeAwareProductNet):
    @ms.jit
    def construct(self, x, y):
        if len(x.shape) == self.rank:
            x = x + y
        return x * y


@ms.jit(capture_mode="bytecode")
def _bytecode_calls_psjit_cell(a, b):
    net = _PsJitShapeAwareProductNet()
    return net(a, b)


def _plain_branching_function(a, b):
    product = a * b
    if len(product.shape) == 2:
        return product
    return product * 2


@ms.jit
def _psjit_branching_helper(a, b):
    return a * b


@ms.jit(capture_mode="bytecode")
def _bytecode_branching_with_psjit(a, b):
    product = _psjit_branching_helper(a, b)
    if len(product.shape) == 2:
        return product
    return product * 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_nested_psjit_method():
    """
    Feature: Combine ms.jit capture modes.
    Description: Execute a bytecode-compiled construct that calls an AST-compiled helper.
    Expectation: JIT mixed-mode result matches pynative result.
    Migrated from: test_pijit_use.py::test_pijit_nest_psjit
    """
    input_np = np.ones((2, 3), np.float32)

    pynative_net = _PlainAddNet()
    pynative_result = pynative_net(Tensor(input_np))

    jit_net = _BytecodeOuterAstInnerNet()
    jit_result = jit_net(Tensor(input_np))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_called_inside_psjit():
    """
    Feature: Combine ms.jit capture modes.
    Description: Execute an AST-compiled construct that calls a bytecode-compiled helper.
    Expectation: JIT mixed-mode result matches pynative result.
    Migrated from: test_pijit_use.py::test_pijit_in_psjit
    """
    input_np = np.ones((2, 3), np.float32)

    pynative_net = _PlainAddNet()
    pynative_result = pynative_net(Tensor(input_np))

    jit_net = _AstOuterBytecodeInnerNet()
    jit_result = jit_net(Tensor(input_np))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_with_single_op_pipeline():
    """
    Feature: ms.jit nested call.
    Description: Call a bytecode-compiled sub-function inside a pynative network with trig operators.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_use.py::test_pijit_with_single_op
    """
    input_np = np.ones((2, 3), np.float32)

    pynative_net = _PlainTrigNet()
    pynative_result = pynative_net(Tensor(input_np))

    jit_net = _BytecodeFuncTrigNet()
    jit_result = jit_net(Tensor(input_np))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_jit_before_psjit_pipeline():
    """
    Feature: ms.jit mixed decorators.
    Description: Call a bytecode-compiled helper before an AST-compiled helper within the same network.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_use.py::test_pijit_before_psjit
    """
    input_np = np.ones((2, 3), np.float32)

    pynative_net = _PlainTwoStageTrigNet()
    pynative_result = pynative_net(Tensor(input_np))

    jit_net = _MixedDecoratorTrigNet()
    jit_result = jit_net(Tensor(input_np))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_function_calls_psjit_cell():
    """
    Feature: Combine bytecode and default jit capture modes.
    Description: Call a bytecode-compiled function that instantiates a psjit-compiled cell handling rank-based branching.
    Expectation: JIT mixed-mode result matches pynative result.
    Migrated from: test_pijit_ai4sci.py::test_pijit_ai4sci_nest_psjit_net
    """
    input_np = np.array([[1, 2], [3, 4]], np.float32)
    other_np = np.array([[1, 2], [3, 4]], np.float32)

    pynative_net = _PlainShapeAwareProductNet()
    pynative_result = pynative_net(Tensor(input_np), Tensor(other_np))

    jit_result = _bytecode_calls_psjit_cell(Tensor(input_np), Tensor(other_np))

    match_array(pynative_result, jit_result, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bytecode_function_calls_psjit_function():
    """
    Feature: Combine bytecode and default jit capture modes.
    Description: Call a bytecode-compiled function that invokes a psjit helper and applies branching based on tensor rank.
    Expectation: JIT mixed-mode result matches pynative result.
    Migrated from: test_pijit_ai4sci.py::test_pijit_ai4sci_in_pijit_func
    """
    input_np = np.array([[1, 2], [3, 4]], np.float32)
    other_np = np.array([[1, 2], [3, 4]], np.float32)

    pynative_result = _plain_branching_function(Tensor(input_np), Tensor(other_np))

    jit_result = _bytecode_branching_with_psjit(Tensor(input_np), Tensor(other_np))

    match_array(pynative_result, jit_result, error=5)
