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
"""Test @jit(fullgraph=True)"""
import numpy as np
import pytest

from mindspore import Tensor, ops, context, jit
from mindspore.nn import Cell
from mindspore.graph._utils.pijit_context import Unsupported

from tests.st.pi_jit.share.utils import pi_jit_with_config, match_array
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_1():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True, and there's a graph-break in tested function.
    Expectation: Throw exception.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(x: Tensor):
        x = x + 1
        print('Graph break!', flush=True)  # graph break
        return x * 2

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    assert "print('Graph break!', flush=True)" in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_2():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.loop_unrolling=False; 3.there's a for-loop in tested function.
    Expectation: Throw exception.
    """
    jit_cfg = {'loop_unrolling': False}

    @pi_jit_with_config(jit_config=jit_cfg, fullgraph=True)
    def fn(x: Tensor):
        for i in range(3):  # graph break!
            x = x + 1
        return x

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    assert 'for i in range(3):' in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_3():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.there are unsupported bytecodes(LOAD_BUILD_CLASS, LOAD_CLASSDEREF).
    Expectation: Throw exception.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn():
        x = 42

        class Inner:  # LOAD_BUILD_CLASS, unsupported bytecode.
            y = x  # x is freevar

        return Inner

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(Unsupported) as info:
        o = fn()
    err_msg = str(info.value)
    assert 'class Inner:' in err_msg
    assert 'Hint: ByteCode LOAD_BUILD_CLASS is not supported' in err_msg
    assert 'Hint: See https://docs.python.org/3/library/dis.html for bytecode semantics' in err_msg


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_4():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True, and there's a graph-break in inner function.
    Expectation: Throw exception.
    """

    def inner():
        print('Graph break in inner function!', flush=True)  # graph break

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(x: Tensor):
        x = x + 1
        inner()
        return x * 2

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    err_msg = str(info.value)
    assert "inner()" in err_msg
    assert "print('Graph break in inner function!', flush=True)" in err_msg
    assert err_msg.find("inner()") < err_msg.find("print('Graph break in inner function!', flush=True)")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_5():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.unsupported iterable type.
    Expectation: Throw exception.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(seq: set):
        ret = 0
        for i in seq:  # graph break!
            ret = ret + i
        return ret

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(Unsupported) as info:
        s = {1, 2, 3}
        o = fn(s)
    err_msg = str(info.value)
    assert 'for i in seq:' in err_msg
    assert 'Hint: Unsupported iterable type: set' in err_msg


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_and_compile_with_try_True():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.compile_with_try=True.
    Expectation: Throw exception.
    """

    @pi_jit_with_config(jit_config={'compile_with_try': True}, fullgraph=True)
    def fn(x: Tensor):
        x = x + 1
        print('Graph break!', flush=True)  # graph break
        return x * 2

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    assert "print('Graph break!', flush=True)" in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_no_graph_break():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with no graph break, should execute successfully.
    Expectation: JIT forward and gradient results match pynative.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_001
    """
    from tests.st.pi_jit.share.grad import GradOfFirstInput

    class Model(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            y = x + x
            z = x * x
            out = ops.div(y, z)
            return out * self.a

    x = Tensor(np.ones((2, 3), np.float32))

    # Pynative mode: forward + gradient
    net = Model()
    pynative_result = net(x)
    grad_net = GradOfFirstInput(net, sens_param=True)
    grad_net.set_train()
    output_grad = Tensor(np.random.randn(*pynative_result.shape).astype(np.float32))
    pynative_grad = grad_net(x, output_grad)

    # JIT mode with fullgraph=True: forward + gradient
    jit_net = Model()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode', fullgraph=True)
    jit_result = jit_net(x)
    jit_grad_net = GradOfFirstInput(jit_net, sens_param=True)
    jit_grad_net.set_train()
    jit_grad = jit_grad_net(x, output_grad)

    # Compare forward results and gradients
    match_array(pynative_result, jit_result)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_at_start():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True, and there's a graph break at the beginning of the function.
    Expectation: Throw Unsupported exception with correct error message.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_002
    """

    class Net2(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        @jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x):
            x = x.asnumpy()  # graph break
            y = x * x
            z = Tensor(y)
            out = ops.div(z, z)
            return out * self.a

    x = Tensor(np.random.rand(2, 3).astype(np.float32))

    net = Net2()
    with pytest.raises(Unsupported) as e:
        net(x)
    assert "x = x.asnumpy()" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_at_middle():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True, and there's a graph break later in the function.
    Expectation: Throw Unsupported exception with correct error message.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_003
    """

    class Net2(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        @jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x):
            x = (x + x) * self.a
            y = x.asnumpy()  # graph break
            return Tensor(y)

    x = Tensor(np.random.rand(2, 3).astype(np.float32))

    net = Net2()
    with pytest.raises(Unsupported) as e:
        net(x)
    assert "y = x.asnumpy()" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_multiple_breaks_and_only_first_is_reported():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with multiple graph break points, only the first one is reported.
    Expectation: Throw Unsupported exception showing only the first graph break.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_004
    """

    class Net4(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        @jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x):
            x = x.asnumpy()  # first graph break
            y = x * x
            z = Tensor(y)
            out = ops.div(z, z) * self.a
            return Tensor(out.asnumpy())  # second graph break (not reported)

    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    net = Net4()

    with pytest.raises(Unsupported) as e:
        net(x)
    assert "x = x.asnumpy()" in str(e.value)
    assert "out.asnumpy()" not in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_in_for_if():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with graph break inside control flow (for-if).
    Expectation: Throw Unsupported exception with correct error message.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_005
    """

    class Net5(Cell):
        def __init__(self):
            super().__init__()
            self.a = 2

        @jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x):
            out = x
            for i in range(4):
                if i < self.a:
                    out = out + Tensor(x.asnumpy() * 2)  # graph break
                else:
                    out = out - x
            return out

    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    net = Net5()
    with pytest.raises(Unsupported) as e:
        net(x)
    assert "out + Tensor(x.asnumpy() * 2)" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_in_method_call():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with graph break in a Cell instance method called from construct.
    Expectation: Throw Unsupported exception with call stack information.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_006
    """

    class Net6(Cell):
        def __init__(self):
            super().__init__()
            self.k = 3

        def func(self, x):
            x = x.asnumpy()  # graph break
            flag = bool((x > self.k).all())
            return flag

        @jit(capture_mode="bytecode", fullgraph=True)
        def construct(self, x):
            if self.func(x):
                out = x + x
            else:
                out = x * x
            return out

    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    net = Net6()
    with pytest.raises(Unsupported) as e:
        net(x)
    assert "if self.func(x):" in str(e.value)
    assert "x = x.asnumpy()" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_in_nested_function():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with graph break in a nested function call.
    Expectation: Throw Unsupported exception with call stack information.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_007
    """

    @jit(capture_mode="bytecode", fullgraph=True)
    def f1(x):
        x = x * 2
        y = f2(x)
        return x + y

    def f2(x):
        x = x - 1
        x = Tensor(x.asnumpy())  # graph break
        return x * 2

    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    with pytest.raises(Unsupported) as e:
        f1(x)
    assert " y = f2(x)" in str(e.value)
    assert "x = Tensor(x.asnumpy())" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_three_layer_nesting():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with graph break in three-layer nested function calls.
    Expectation: Throw Unsupported exception showing all layers in call stack.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_008
    """

    @jit(capture_mode="bytecode", fullgraph=True)
    def f1(x):
        x = f2(x)
        return ops.neg(x)

    def f2(x):
        x = f3(x)
        return ops.neg(x)

    def f3(x):
        x = x + 1
        x.asnumpy()  # graph break
        return ops.neg(x)

    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    with pytest.raises(Unsupported) as e:
        f1(x)
    assert "x = f2(x)" in str(e.value)
    assert "x = f3(x)" in str(e.value)
    assert "x.asnumpy()" in str(e.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fullgraph_break_in_if_condition():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True with graph break in an if condition expression.
    Expectation: Throw Unsupported exception with correct error message.
    Legacy: test_parse_pijit_fullgraph.py::test_parse_pijit_fullgraph_009
    """

    @jit(capture_mode="bytecode", fullgraph=True)
    def f1(x):
        x = f2(x)
        return ops.relu(x)

    def f2(x):
        if Tensor(x.asnumpy()) > 2:  # graph break
            x = x + f3(x)
        return ops.relu(x)

    def f3(x):
        x = x + 1
        x.asnumpy()
        return ops.relu(x)

    x = Tensor(np.random.rand(2, 3).astype(np.float32))
    with pytest.raises(Unsupported) as e:
        f1(x)
    assert "x = f2(x)" in str(e.value)
    assert "if Tensor(x.asnumpy()) > 2:" in str(e.value)
