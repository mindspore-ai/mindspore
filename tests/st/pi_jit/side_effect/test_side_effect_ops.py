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
"""Test side effect operation in pijit"""
import os
import pytest
import sys
import time
import tempfile
from contextlib import contextmanager

import mindspore
from tests.mark_utils import arg_mark
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, jit, ops, context
from mindspore.ops.composite import GradOperation
from tests.st.pi_jit.share.utils import assert_no_graph_break, assert_executed_by_graph_mode, match_array
from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num


class Capture():
    def __init__(self):
        self._old_stdout = sys.stdout
        self._stdout_fd = sys.stdout.fileno()
        self._saved_stdout_fd = os.dup(sys.stdout.fileno())
        self._file = tempfile.TemporaryFile(mode='w+t')
        self.output = ''

    def start(self):
        os.dup2(self._file.fileno(), self._stdout_fd)

    def stop(self):
        os.dup2(self._saved_stdout_fd, self._stdout_fd)
        os.close(self._saved_stdout_fd)
        sys.stdout = self._old_stdout
        self._file.seek(0)
        self.output = self._file.read()
        self._file.close()


@contextmanager
def capture(cap):
    cap.start()
    try:
        yield cap
    finally:
        cap.stop()


def check_output(output, patterns):
    assert output, "Capture output failed!"
    index = 0
    for pattern in patterns:
        index = output.find(pattern, index)
        assert index != -1, "Unexpected output:\n" + output + "\n--- pattern ---\n" + pattern


def count_output(output, target, num):
    assert output, "Capture output failed!"
    assert output.count(target) == num


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_tensor():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = ops.Print()

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x):
            self.print("result: ", x + 1)
            return x + 1

    context.set_context(mode=context.PYNATIVE_MODE)

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        net = Net()
        out = net(input_x)
        assert np.all(out.asnumpy() == np.array([4]))
        assert_no_graph_break(Net.construct)
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['result: ', 'Tensor(shape=[], dtype=Int32, value=4)']
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_tensor_multiple_times():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = ops.Print()

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x):
            self.print("result1: ", x)
            x = x + 1
            self.print("result2: ", x)
            return x

    context.set_context(mode=context.PYNATIVE_MODE)

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        net = Net()
        out = net(input_x)
        assert np.all(out.asnumpy() == np.array([4]))
        assert_no_graph_break(Net.construct)
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['result1: ', 'Tensor(shape=[], dtype=Int32, value=3)',
                'result2: ', 'Tensor(shape=[], dtype=Int32, value=4)']
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_constant_scalar():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = ops.Print()

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x, y):
            self.print("constant: ", y)
            self.print("result1: ", x)
            x = x + y
            self.print("result2: ", x)
            return x

    context.set_context(mode=context.PYNATIVE_MODE)

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = 1
        net = Net()
        out = net(input_x, input_y)
        assert np.all(out.asnumpy() == np.array([4]))
        assert_no_graph_break(Net.construct)
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['constant: ', '1',
                'result1: ', 'Tensor(shape=[], dtype=Int32, value=3)',
                'result2: ', 'Tensor(shape=[], dtype=Int32, value=4)']
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_in_sub_graph():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def __init__(self):
            super(InnerNet, self).__init__()
            self.print = ops.Print()

        def construct(self, x, y):
            self.print("inner constant: ", y)
            self.print("inner result1: ", x)
            x = x + y
            self.print("inner result2: ", x)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = ops.Print()
            self.inner_net = InnerNet()

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x, y):
            ret = self.inner_net(x, y)
            self.print("out result: ", ret)
            return ret

    context.set_context(mode=context.PYNATIVE_MODE)

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = 1
        net = Net()
        out = net(input_x, input_y)
        assert np.all(out.asnumpy() == np.array([4]))
        assert_no_graph_break(Net.construct)
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['inner constant: ', '1',
                'inner result1: ', 'Tensor(shape=[], dtype=Int32, value=3)',
                'inner result2: ', 'Tensor(shape=[], dtype=Int32, value=4)',
                'out result: ', 'Tensor(shape=[], dtype=Int32, value=4)', ]
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_print_in_sub_graph_with_no_return():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """

    class InnerNet(nn.Cell):
        def __init__(self):
            super(InnerNet, self).__init__()
            self.print = ops.Print()

        def construct(self, x, y):
            self.print("inner constant: ", y)
            self.print("inner result1: ", x)
            x = x + y
            self.print("inner result2: ", x)

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.print = ops.Print()
            self.inner_net = InnerNet()

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x, y):
            self.inner_net(x, y)
            return x + y

    context.set_context(mode=context.PYNATIVE_MODE)

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = 1
        net = Net()
        out = net(input_x, input_y)
        assert np.all(out.asnumpy() == np.array([4]))
        assert_no_graph_break(Net.construct)
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['inner constant: ', '1',
                'inner result1: ', 'Tensor(shape=[], dtype=Int32, value=3)',
                'inner result2: ', 'Tensor(shape=[], dtype=Int32, value=4)']
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_base_grad_operation_with_side_effect():
    """
    Feature: One stage GradOperation
    Description: Test One stage GradOperation with no graph break
    Expectation: No exception.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            ret = x + y
            print("x + y: ", ret)
            return ret

    class GradNet(nn.Cell):
        def __init__(self, net, ):
            super(GradNet, self).__init__()
            self.net = net
            self.grad_op = GradOperation(False, False, False)

        @pi_jit_with_config(jit_config={"compile_with_try": False})
        def construct(self, x, y):
            grad_ret = self.grad_op(self.net)(x, y)
            return grad_ret

    context.set_context(mode=context.PYNATIVE_MODE)
    cap = Capture()
    with capture(cap):
        net = Net()
        grad_net = GradNet(net)
        a = Tensor([1])
        b = Tensor([2, ])
        grad_net(a, b)
        sys.stdout.flush()
        time.sleep(2.0)

    count_output(cap.output, "x + y", 1)


@pytest.mark.skip(reason="The implementation of InplaceCopy primitive has been changed. Fix it later.")
@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_Tensor_inplace_copy_v1():
    """
    Feature: test Tensor.copy_().
    Description: Tensor.copy_() is a memory side-effect op.
    Expectation: no graph break.
    """

    context.set_context(mode=context.PYNATIVE_MODE)

    def fn(x: Tensor, y: Tensor):
        y = y + 1
        x.copy_(y)
        return y

    x1 = mindspore.tensor([1, 2, 3])
    y1 = mindspore.tensor([2, 3, 3])
    o1 = fn(x1, y1)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    x2 = mindspore.tensor([1, 2, 3])
    y2 = mindspore.tensor([2, 3, 3])
    o2 = compiled_fn(x2, y2)

    match_array(o1, o2)
    match_array(x1, x2)
    assert_executed_by_graph_mode(compiled_fn)
    check_ir_num('graph_before_compile', 1)


@pytest.mark.skip(reason="The implementation of InplaceCopy primitive has been changed. Fix it later.")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_Tensor_inplace_copy_v2():
    """
    Feature: test Tensor.copy_().
    Description: Tensor.copy_() is a memory side-effect op, it is called in a subgraph.
    Expectation: no graph break.
    """

    def f2(x, y):
        y = y + 1
        x.copy_(y)

    def fn(x: Tensor, y: Tensor):
        f2(x, y)
        return y * 2

    x1 = mindspore.tensor([1, 2, 3])
    y1 = mindspore.tensor([2, 3, 3])
    o1 = fn(x1, y1)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    x2 = mindspore.tensor([1, 2, 3])
    y2 = mindspore.tensor([2, 3, 3])
    o2 = compiled_fn(x2, y2)

    match_array(o1, o2)
    match_array(x1, x2)
    assert_executed_by_graph_mode(compiled_fn)


@pytest.mark.skip(reason="The implementation of InplaceCopy primitive has been changed. Fix it later.")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_Tensor_inplace_copy_v3():
    """
    Feature: test Tensor.copy_().
    Description: Tensor.copy_() is a memory side-effect op, it is called in a subgraph.
    Expectation: no graph break.
    """

    def f2(x, y):
        y = y + 1
        x.copy_(y)
        return y, y

    def fn(x: Tensor, y: Tensor):
        out = f2(x, y)
        y2 = out[0]  # y2 is not used.
        return y * 2

    x1 = mindspore.tensor([1, 2, 3])
    y1 = mindspore.tensor([2, 3, 3])
    o1 = fn(x1, y1)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    x2 = mindspore.tensor([1, 2, 3])
    y2 = mindspore.tensor([2, 3, 3])
    o2 = compiled_fn(x2, y2)

    match_array(o1, o2)
    match_array(x1, x2)
    assert_executed_by_graph_mode(compiled_fn)
