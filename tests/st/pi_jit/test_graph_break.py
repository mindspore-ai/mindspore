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
"""Test pijit graph break"""

import numpy as np
from mindspore import Tensor, jit, ops
from mindspore.nn import Cell

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.grad import compute_grad_of_net_inputs
from tests.st.pi_jit.share.utils import match_array


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_all_graph_ops():
    """
    Feature: PIJit graph break handling.
    Description: Pure graph operators without graph break should keep gradients identical between modes.
    Expectation: Forward output and gradient match between pynative and JIT.
    Migrated from: test_pijit_graph_split.py::test_pijit_all_graph
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            y = x + x
            z = x * x
            out = ops.div(y, z)
            return out * self.a

    input_np = np.ones((2, 3), np.float32)

    pynative_net = Net()
    pynative_net.set_grad()
    pynative_out = pynative_net(Tensor(input_np))

    sens_np = np.random.randn(*pynative_out.shape).astype(np.float32)
    pynative_sens = Tensor(sens_np)
    pynative_grad = compute_grad_of_net_inputs(pynative_net, Tensor(input_np), sens=pynative_sens)

    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(Tensor(input_np))
    jit_sens = Tensor(sens_np)
    jit_grad = compute_grad_of_net_inputs(jit_net, Tensor(input_np), sens=jit_sens)

    match_array(pynative_out, jit_out)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_all_numpy_ops():
    """
    Feature: PIJit graph break handling.
    Description: Entire construct body uses numpy operations after converting inputs to numpy.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_graph_split.py::test_pijit_all_pynative
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x, y):
            a, b = x.asnumpy(), y.asnumpy()
            c = np.squeeze(b)
            d = np.stack([a, c])
            return Tensor(d) * self.a

    input_x = Tensor(np.ones((2, 3), np.float32))
    input_y = Tensor(np.ones((2, 1, 3, 1), np.float32))
    pynative_net = Net()
    pynative_out = pynative_net(input_x, input_y)

    jit_x = Tensor(np.ones((2, 3), np.float32))
    jit_y = Tensor(np.ones((2, 1, 3, 1), np.float32))
    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(jit_x, jit_y)

    match_array(pynative_out, jit_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_numpy_then_graph():
    """
    Feature: PIJit graph break handling.
    Description: The construct function switches from graph mode to numpy and back once.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_graph_split.py::test_pijit_graph_split_1
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            x_np = x.asnumpy()
            y = x_np * x_np
            z = Tensor(y)
            out = ops.div(z, z)
            return out * self.a

    input_np = np.ones((2, 3), np.float32)

    pynative_net = Net()
    pynative_out = pynative_net(Tensor(input_np))

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(Tensor(input_np))

    match_array(pynative_out, jit_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_graph_then_numpy():
    """
    Feature: PIJit graph break handling.
    Description: Graph computation happens first, followed by numpy conversion at the end.
    Expectation: JIT result matches pynative result.
    Migrated from: test_pijit_graph_split.py::test_pijit_graph_split_2
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            x = (x + x) * self.a
            x_np = x.asnumpy()
            return Tensor(x_np)

    input_np = np.ones((2, 3), np.float32)

    pynative_net = Net()
    pynative_out = pynative_net(Tensor(input_np))

    jit_net = Net()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(Tensor(input_np))

    match_array(pynative_out, jit_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_numpy_graph_numpy():
    """
    Feature: PIJit graph break handling.
    Description: Construct interleaves numpy and graph computations twice.
    Expectation: Forward output and gradient match between pynative and JIT.
    Migrated from: test_pijit_graph_split.py::test_pijit_graph_split_3
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            x_np = x.asnumpy()
            y = x_np * x_np
            z = Tensor(y)
            out = ops.div(z, z) * self.a
            return Tensor(out.asnumpy())

    input_np = np.ones((2, 3), np.float32)

    pynative_net = Net()
    pynative_net.set_grad()
    pynative_out = pynative_net(Tensor(input_np))

    sens_np = np.random.randn(*pynative_out.shape).astype(np.float32)
    pynative_sens = Tensor(sens_np)
    pynative_grad = compute_grad_of_net_inputs(pynative_net, Tensor(input_np), sens=pynative_sens)

    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(Tensor(input_np))
    jit_sens = Tensor(sens_np)
    jit_grad = compute_grad_of_net_inputs(jit_net, Tensor(input_np), sens=jit_sens)

    match_array(pynative_out, jit_out)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_graph_numpy_graph():
    """
    Feature: PIJit graph break handling.
    Description: Graph computation, numpy fallback, and graph resume execute in sequence.
    Expectation: Forward output and gradient match between pynative and JIT.
    Migrated from: test_pijit_graph_split.py::test_pijit_graph_split_4
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 1

        def construct(self, x):
            x = (x + x) * self.a
            x_np = x.asnumpy()
            y = x_np * x_np
            z = Tensor(y)
            out = ops.div(z, z)
            return out

    input_np = np.ones((2, 3), np.float32)

    pynative_net = Net()
    pynative_net.set_grad()
    pynative_out = pynative_net(Tensor(input_np))

    sens_np = np.random.randn(*pynative_out.shape).astype(np.float32)
    pynative_sens = Tensor(sens_np)
    pynative_grad = compute_grad_of_net_inputs(pynative_net, Tensor(input_np), sens=pynative_sens)

    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(Tensor(input_np))
    jit_sens = Tensor(sens_np)
    jit_grad = compute_grad_of_net_inputs(jit_net, Tensor(input_np), sens=jit_sens)

    match_array(pynative_out, jit_out)
    match_array(pynative_grad, jit_grad, error=5)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_graph_break_inside_loop():
    """
    Feature: PIJit graph break handling.
    Description: Graph break occurs inside a for-loop when converting tensors to numpy.
    Expectation: Forward output and gradient match between pynative and JIT.
    Migrated from: test_pijit_graph_split.py::test_pijit_graph_split_for
    """

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.a = 2

        def construct(self, x):
            out = x
            for i in range(4):
                if i < self.a:
                    out = out + Tensor(x.asnumpy() * 2)
                else:
                    out = out - x
            return out

    input_np = np.ones((2, 3), np.float32)

    pynative_net = Net()
    pynative_net.set_grad()
    pynative_out = pynative_net(Tensor(input_np))

    sens_np = np.random.randn(*pynative_out.shape).astype(np.float32)
    pynative_sens = Tensor(sens_np)
    pynative_grad = compute_grad_of_net_inputs(pynative_net, Tensor(input_np), sens=pynative_sens)

    jit_net = Net()
    jit_net.set_grad()
    jit_net.construct = jit(jit_net.construct, capture_mode='bytecode')
    jit_out = jit_net(Tensor(input_np))
    jit_sens = Tensor(sens_np)
    jit_grad = compute_grad_of_net_inputs(jit_net, Tensor(input_np), sens=jit_sens)

    match_array(pynative_out, jit_out)
    match_array(pynative_grad, jit_grad, error=5)
