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
""" test trace functions """

import sys
import time
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.compiler.auto_monad.capture import Capture, capture, check_output
import mindspore as ms
from mindspore.graph.api import _pynative_executor
from mindspore.graph.jit_begin_end import _jit_begin as jit_begin
from mindspore.graph.jit_begin_end import _jit_end as jit_end
from mindspore.ops import functional as F
import mindspore.common.dtype as mstype


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_1():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == 12
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == 12


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_2():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return x, y, z

    trace_net = TraceNet()
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == (1, 3, 12)
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == (1, 3, 12)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_3():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(x, y, z):
        return x + y + z

    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == 6
    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == 6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_4():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(inputs):
        x, y, z = inputs
        return x + y + z

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    res = foo(inputs)
    assert res == 6
    res = foo(inputs)
    assert res == 6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_5():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(inputs):
        x, y, z = inputs
        return x, y, z, x + y + z

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    res = foo(inputs)
    assert res == (1, 2, 3, 6)
    res = foo(inputs)
    assert res == (1, 2, 3, 6)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_6():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(inputs):
        x = ms.ops.addn(inputs)
        return x, inputs

    inputs = ms.mutable((ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)))
    res = foo(inputs)
    assert res == (6, (1, 2, 3))
    res = foo(inputs)
    assert res == (6, (1, 2, 3))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_7():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(inputs):
        x = ms.ops.addn(inputs)
        return x, inputs

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    res = foo(inputs)
    assert res == (6, [1, 2, 3])
    res = foo(inputs)
    assert res == (6, [1, 2, 3])


class TensorGetItem(ms.nn.Cell):
    def construct(self, tensor, index):
        res = tensor[index]
        return res


class TensorSetItem(ms.nn.Cell):
    def construct(self, tensor, index, value):
        tensor[index] = value


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_trace_tensor_tuple_index_0():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(x, index):
        return TensorGetItem()(x, index)

    input_1d_np = np.array([1]).astype(np.float32)
    input_1d_ms = ms.Tensor(input_1d_np, ms.dtype.float32)
    input_3d_np = np.random.randint(3, size=(3, 4, 5)).astype(np.int32)
    input_3d_ms = ms.Tensor(input_3d_np, ms.dtype.float32)

    index_np_1 = (0,)
    index_np_2 = (1, 2)
    index_np_3 = (1, 2, 3)
    index_np_4 = (3, 4, 4)
    index_np_5 = (1, 2, 3, 4)

    output_1d_ms = foo(input_1d_ms, index_np_1)
    print(f'output_1d_ms: {output_1d_ms}')
    output_3d_ms = foo(input_3d_ms, index_np_3)
    print(f'output_3d_ms: {output_3d_ms}')
    assert output_1d_ms == input_1d_np.item(index_np_1)
    assert output_3d_ms == input_3d_np.item(index_np_3)

    with pytest.raises(IndexError):
        foo(input_1d_ms, index_np_2)
        _pynative_executor.sync()

    with pytest.raises(ValueError):
        foo(input_3d_ms, index_np_2)
        _pynative_executor.sync()

    with pytest.raises(IndexError):
        foo(input_3d_ms, index_np_4)
        _pynative_executor.sync()

    with pytest.raises(IndexError):
        foo(input_3d_ms, index_np_5)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_side_effect_1():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            ms.ops.Print()(a)
            z = x + a
            ms.ops.Print()(z)
            z = z + self.x
            ms.ops.Print()(z)
            z = z * y
            ms.ops.Print()(z)
            return z

    trace_net = TraceNet()
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == 12

    cap = Capture()
    with capture(cap):
        res = trace_net(ms.Tensor(1), ms.Tensor(3))
        sys.stdout.flush()
        time.sleep(2.0)
    assert res == 12
    patterns = {'Tensor(shape=[], dtype=Int64, value=2)\n'
                'Tensor(shape=[], dtype=Int64, value=3)\n'
                'Tensor(shape=[], dtype=Int64, value=4)\n'
                'Tensor(shape=[], dtype=Int64, value=12)'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_side_effect_2():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(inputs):
        x = ms.ops.addn(inputs)
        ms.ops.Print()('x: ', x)
        return x, inputs

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    res = foo(inputs)
    assert res == (6, [1, 2, 3])

    cap = Capture()
    with capture(cap):
        res = foo(inputs)
        sys.stdout.flush()
        time.sleep(2.0)
    assert res == (6, [1, 2, 3])
    patterns = {'x: \nTensor(shape=[], dtype=Int64, value=6)'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_begin_end_1():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    def foo(x, y, z):
        a = ms.Tensor(2)
        z = z + a
        jit_begin("__trace__jit_block__1__", x, y, z)
        z = z + x
        z = z * y
        z = jit_end(z)
        return z

    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == 12
    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == 12


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_begin_end_2():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    def foo(x, y, z):
        t = ms.Tensor(2)
        z = z + t
        jit_begin("__trace__jit_block__2__", x, y, z)
        a = z + x
        b = a * y
        a, b = jit_end(a, b)
        return a, b

    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == (6, 12)
    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == (6, 12)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_begin_end_3():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    def bar(x, y, z):
        t = ms.Tensor(2)
        z = z + t
        # <-- Start func graph building.
        jit_begin("__trace__jit_block__3__", x, y, z)
        a = z + x
        b = a * y
        return a, b

    def foo(x, y, z):
        inputs = bar(x, y, z)
        a = ms.ops.addn(inputs)
        a = jit_end(a)  # <-- End func graph building.
        b = a * y
        return a, b

    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == (18, 36)
    res = foo(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == (18, 36)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_1():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    def jit_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="ast")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return trace_func(a, x, y, self.x)
            return jit_func(a, x, y, self.x)

    net = Net()
    res_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_jit = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_trace == res_jit


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_2():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return x, y, z

    def jit_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return x, y, z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="ast")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return trace_func(a, x, y, self.x)
            return jit_func(a, x, y, self.x)

    net = Net()
    res_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_jit = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_trace == res_jit


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_3():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def trace_func(inputs):
        x, y, z = inputs
        return x, y, z, x + y + z

    def jit_func(inputs):
        x, y, z = inputs
        return x, y, z, x + y + z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="ast")
        def construct(self, inputs, flag):
            if flag:
                return trace_func(inputs) + self.x
            return jit_func(inputs) + self.x

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    net = Net()
    res_trace = net(inputs, True)
    res_jit = net(inputs, False)
    assert np.allclose(res_trace.asnumpy(), res_jit.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_side_effect_1():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(x, y, self_x, a):
        z = x + a
        ms.ops.Print()(z)
        z = z + self_x
        ms.ops.Print()(z)
        z = z * y
        ms.ops.Print()(z)
        return z

    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="ast")
        def construct(self, x, y):
            a = ms.Tensor(2)
            ms.ops.Print()(a)
            return foo(x, y, self.x, a)

    trace_net = TraceNet()
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == 12

    cap = Capture()
    with capture(cap):
        res = trace_net(ms.Tensor(1), ms.Tensor(3))
        sys.stdout.flush()
        time.sleep(2.0)
    assert res == 12
    patterns = {'Tensor(shape=[], dtype=Int64, value=2)\n'
                'Tensor(shape=[], dtype=Int64, value=3)\n'
                'Tensor(shape=[], dtype=Int64, value=4)\n'
                'Tensor(shape=[], dtype=Int64, value=12)'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_4():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """
    @ms.jit(capture_mode="ast")
    def foo(x, y):
        return x * y

    @ms.jit(capture_mode="trace")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = foo(z, y)
        return x, y, z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="ast")
        def construct(self, x, y):
            a = ms.Tensor(2)
            return trace_func(a, x, y, self.x)

    net = Net()
    with pytest.raises(RuntimeError) as err:
        net(ms.Tensor(1), ms.Tensor(3))
    assert "Please check the current code for its nesting usage." in str(
        err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_ast_1():
    """
    Feature: JIT ast function nested by JIT trace
    Description: JIT ast function nested by JIT trace
    Expectation: No exception
    """
    @ms.jit(capture_mode="ast")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    def jit_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return trace_func(a, x, y, self.x)
            return jit_func(a, x, y, self.x)

    net = Net()
    res_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_jit = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_trace == res_jit


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_ast_2():
    """
    Feature: JIT ast function nested by JIT trace
    Description: JIT ast function nested by JIT trace
    Expectation: No exception
    """
    @ms.jit(capture_mode="ast")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return x, y, z

    def jit_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return x, y, z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return trace_func(a, x, y, self.x)
            return jit_func(a, x, y, self.x)

    net = Net()
    res_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_jit = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_trace == res_jit


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_ast_3():
    """
    Feature: JIT ast function nested by JIT trace
    Description: JIT ast function nested by JIT trace
    Expectation: No exception
    """
    @ms.jit(capture_mode="ast")
    def trace_func(inputs):
        x, y, z = inputs
        return x, y, z, x + y + z

    def jit_func(inputs):
        x, y, z = inputs
        return x, y, z, x + y + z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, inputs, flag):
            if flag:
                return trace_func(inputs) + self.x
            return jit_func(inputs) + self.x

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    net = Net()
    res_trace = net(inputs, True)
    res_jit = net(inputs, False)
    assert np.allclose(res_trace.asnumpy(), res_jit.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_ast_4():
    """
    Feature: JIT ast function nested by JIT trace
    Description: JIT ast function nested by JIT trace
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(x, y):
        return x * y

    @ms.jit(capture_mode="ast")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = foo(z, y)
        return x, y, z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            return trace_func(a, x, y, self.x)

    net = Net()
    with pytest.raises(RuntimeError) as err:
        net(ms.Tensor(1), ms.Tensor(3))
    assert "Please check the current code for its nesting usage." in str(
        err.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_double_trace():
    """
    Feature: JIT trace function nested by JIT trace
    Description: JIT trace function nested by JIT trace
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    def func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return trace_func(a, x, y, self.x)
            return func(a, x, y, self.x)

    net = Net()
    res_double_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_trace = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_double_trace == res_trace


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_5():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        @ms.jit(capture_mode="trace")
        def construct(self, inputs):
            x, y, z = inputs
            return x, y, z, x + y + z

    class JitNet(ms.nn.Cell):
        @ms.jit(capture_mode="ast")
        def construct(self, inputs):
            x, y, z = inputs
            return x, y, z, x + y + z

    class Net(ms.nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.x = ms.Tensor(1)
            self.net = net

        @ms.jit(capture_mode="ast")
        def construct(self, inputs):
            return self.net(inputs) + self.x

    inputs = ms.mutable([ms.Tensor(1), ms.Tensor(2), ms.Tensor(3)])
    jit_inner_net = JitNet()
    trace_inner_net = TraceNet()
    jit_net = Net(jit_inner_net)
    trace_net = Net(trace_inner_net)
    res_trace = trace_net(inputs)
    res_jit = jit_net(inputs)
    assert np.allclose(res_trace.asnumpy(), res_jit.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_tensor_type():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            if isinstance(F.typeof(a), mstype.TensorType):
                z = x + a
                z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == 12
    res = trace_net(ms.Tensor(1), ms.Tensor(3))
    assert res == 12


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_6():
    """
    Feature: JIT trace function nested by JIT ast
    Description: JIT trace function nested by JIT ast
    Expectation: No exception
    """

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def trace_func(self, a, x, y, self_x):
            z = x + a
            z = z + self_x
            z = z * y
            return z

        def jit_func(self, a, x, y, self_x):
            z = x + a
            z = z + self_x
            z = z * y
            return z

        @ms.jit(capture_mode="ast")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return self.trace_func(a, x, y, self.x)
            return self.jit_func(a, x, y, self.x)

    net = Net()
    res_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_jit = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_trace == res_jit


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_functional():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    def foo(x, y, z):
        return x + y + z

    foo1 = ms.jit(foo, capture_mode="trace")
    res = foo1(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == 6
    res = foo1(ms.Tensor(1), ms.Tensor(2), ms.Tensor(3))
    assert res == 6


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_empty_input():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo():
        return ms.Tensor(1) + ms.Tensor(2)

    res = foo()
    assert res == 3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_8():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def foo(inputs):
        return inputs[0] + inputs[1] + inputs[2]

    inputs = ms.mutable({0: ms.Tensor(1), 1: ms.Tensor(2), 2: ms.Tensor(3)})
    res = foo(inputs)
    assert res == 6
    res = foo(inputs)
    assert res == 6


@arg_mark(plat_marks=["platform_ascend", "platform_gpu"], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_trace_setitem():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace", backend="ms_backend")
        def construct(self, x, y):
            a = ms.Tensor(2)
            x[0] = y[2]
            z = x + a
            z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    res = trace_net(ms.Tensor([1, 2, 3]), ms.Tensor([4, 5, 6]))
    expected = ms.Tensor([36, 25, 36])
    assert np.allclose(res.asnumpy(), expected.asnumpy())
    res = trace_net(ms.Tensor([1, 2, 3]), ms.Tensor([4, 5, 6]))
    assert np.allclose(res.asnumpy(), expected.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_ops_with_init_inputs():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            shape = (2, 3)
            x[0] = y[2]
            x = ms.ops.broadcast_to(x, shape)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    res = trace_net(ms.Tensor([1, 2, 3]), ms.Tensor([4, 5, 6]))
    expected = ms.Tensor([[36, 25, 36], [36, 25, 36]])
    assert np.allclose(res.asnumpy(), expected.asnumpy())
    res = trace_net(ms.Tensor([1, 2, 3]), ms.Tensor([4, 5, 6]))
    assert np.allclose(res.asnumpy(), expected.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_9():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class DictNet(ms.nn.Cell):
        @ms.jit(capture_mode="trace")
        def construct(self, d):
            x = d.keys()
            y = d.values()
            z = d.items()

            d.update({'e': ms.Tensor(5)})
            q = dict.fromkeys(["one", "two", "three"], 0)
            r = 'b' in d
            d.clear()

            return [x, y, z, q, r, d]

    d = {'a': ms.Tensor(1), 'b': ms.Tensor(2), 'c': ms.Tensor(3)}
    x, y, z, q, r, new_d = DictNet()(d)

    assert list(x) == list(d.keys())
    assert list(y) == list(d.values())
    assert list(z) == list(d.items())
    assert q == dict.fromkeys(["one", "two", "three"], 0) and r
    assert new_d == {}


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nested_trace_with_kwargs():
    """
    Feature: JIT trace function nested by JIT ast with kwargs
    Description: JIT trace function nested by JIT ast with kwargs
    Expectation: No exception
    """
    @ms.jit(capture_mode="trace")
    def trace_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    def jit_func(a, x, y, self_x):
        z = x + a
        z = z + self_x
        z = z * y
        return z

    class Net(ms.nn.Cell):
        def __init__(self):
            super().__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="ast")
        def construct(self, x, y, flag):
            a = ms.Tensor(2)
            if flag:
                return trace_func(a=a, x=x, y=y, self_x=self.x)
            return jit_func(a=a, x=x, y=y, self_x=self.x)

    net = Net()
    res_trace = net(ms.Tensor(1), ms.Tensor(3), True)
    res_jit = net(ms.Tensor(1), ms.Tensor(3), False)
    assert res_trace == res_jit


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_10():
    """
    Feature: JIT trace function
    Description: JIT trace function with ops getting scalar tuple.
    Expectation: No exception
    """
    def get_axis(x):
        shape = ms.ops.shape(x)
        length = ms.ops.tuple_len(shape)
        perm = ms.ops.make_range(0, length)
        return perm

    @ms.jit(capture_mode="trace")
    def func_trace(x):
        x = ms.ops.ReduceMean()(x, get_axis(x))
        return x
    out_trace = func_trace(ms.Tensor([1.0, 2.0, 3.0]))
    assert out_trace == 2
