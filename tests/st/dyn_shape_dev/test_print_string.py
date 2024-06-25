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
""" test print string """
import pytest
import numpy as np
import os
import sys
import random
import string
import time
import tempfile
from contextlib import contextmanager
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, jit, ops
from mindspore.ops import operations as P
from mindspore.ops.primitive import _run_op
from tests.security_utils import security_off_wrap
from tests.st.utils import test_utils


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


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_print(mode):
    """
    Feature: Print string type.
    Description: Print string and verify print result.
    Expectation: No exception and result is correct.
    """
    ms.set_context(mode=mode)
    class Print(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        @jit
        def construct(self, x, y):
            self.print("input_x:", x, "input_y:", y)
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        net = Print()
        out = net(input_x, input_y)
        np.testing.assert_array_equal(out.asnumpy(), input_x.asnumpy())
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['input_x:', 'Tensor(shape=[], dtype=Int32, value=3)',
                'input_y:', 'Tensor(shape=[], dtype=Int32, value=4)']
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_print_addn(mode):
    """
    Feature: Print string type ixed with addn output.
    Description: Print string mixed with addn output and verify print result.
    Expectation: No exception and result is correct.
    """
    ms.set_context(mode=mode)
    class Print(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()
            self.addn = P.AddN()

        def construct(self, x, y):
            t = self.addn([x, y])
            self.print("addn output:", t)
            return t

    cap = Capture()
    with capture(cap):
        input_x = Tensor(3, dtype=ms.int32)
        input_y = Tensor(4, dtype=ms.int32)
        net = Print()
        out = net(input_x, input_y)
        np.testing.assert_array_equal(out.asnumpy(), np.array([7], dtype=np.int32))
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['addn output:', 'Tensor(shape=[], dtype=Int32, value=']
    check_output(cap.output, patterns)


class SideEffectOneInputBprop(nn.Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()
        self.mul = P.Mul()
        self.print1 = P.Print()

    def construct(self, x):
        return self.relu(x)

    def bprop(self, x, out, dout):
        x = self.relu(x)
        x = 5 * x
        self.print1("x1: ", x)
        x = self.mul(x, x)
        self.print1("x2: ", x)
        return (5 * x,)


class GradNetAllInputs(nn.Cell):
    def __init__(self, net):
        super(GradNetAllInputs, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, params1, grad_ys):
        grad_net = self.grad_op(self.net)
        return grad_net(params1, grad_ys)


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_side_effect_bprop_one_input():
    """
    Feature: Test side effect bprop with one input.
    Description: Test side effect bprop with one input.
    Expectation: No exception and result is correct.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    cap = Capture()
    with capture(cap):
        net = SideEffectOneInputBprop()
        grad_net = GradNetAllInputs(net)
        grad_ys = Tensor(np.ones([2, 2]).astype(np.float32))
        input1 = Tensor(np.ones([2, 2]).astype(np.float32))
        grads = grad_net(input1, grad_ys)
        assert grads[0].asnumpy().shape == (2, 2)
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['x1:',
                'Tensor(shape=[2, 2], dtype=Float32, value=',
                '[[ 5.00000000e+00  5.00000000e+00]',
                ' [ 5.00000000e+00  5.00000000e+00]])',
                'x2:',
                'Tensor(shape=[2, 2], dtype=Float32, value=',
                '[[ 2.50000000e+01  2.50000000e+01]',
                ' [ 2.50000000e+01  2.50000000e+01]])',
                ]
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_run_op_print():
    """
    Feature: Test print string by calling _run_op method.
    Description: Test print string by calling _run_op method.
    Expectation: No exception and result is correct.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x):
            _run_op(self.print, "Print", ("TensorStart", x, "TheEnd"))
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor([1, 2, 3])
        net = Net()
        out = net(input_x)
        np.testing.assert_array_equal(out.asnumpy(), np.array([1, 2, 3], dtype=np.int32))
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['TensorStart',
                'Tensor(shape=[3], dtype=Int64, value=[1 2 3])',
                'TheEnd']
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_print_none(mode):
    """
    Feature: Print None and "None".
    Description: Print None and string "None", and verify print result.
    Expectation: No exception and result is correct.
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, y=None):
            if y is not None:
                self.print("y:", y)
            else:
                print("y is", y)
            return y

    cap = Capture()
    with capture(cap):
        net = Net()
        net()
        out2 = net("None")
        assert out2 == "None"
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['y is', 'None',
                'y:', 'None']
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_to_file():
    """
    Feature: Print data to file
    Description: Test print data to file
    Expectation: No exception and print data file was created
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    print_path = './' + ''.join(random.sample(string.ascii_letters + string.digits, 32))
    print_file = f'{print_path}/print.data'
    ms.set_context(print_file_path=print_file)
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x):
            self.print("x:", x)
            return x

    cap = Capture()
    with capture(cap):
        net = Net()
        x = Tensor(np.ones([2, 2], dtype=np.float32))
        net(x)
        time.sleep(0.1)

    assert os.path.exists(print_file)
    os.system(f'rm -rf {print_path}')


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_kbk_control_flow_print_string():
    """
    Feature: Test print string in control flow.
    Description: Print string in control flow, and verify print result.
    Expectation: No exception and result is correct.
    """
    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_context(jit_level='O0')

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.print = P.Print()

        def construct(self, x, y):
            print('network_with_CONTrolZ_flow_0.npy')
            if y > 0:
                print('network_with_CONTrolZ_flow_0.npy')
            return x, y

    cap = Capture()
    with capture(cap):
        net = Net()
        input_x = Tensor(np.random.uniform(0.0, 2.0, size=[2, 1]).astype(np.int8))
        _, out = net(input_x, Tensor([1]))
        assert out.asnumpy() == 1
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = ['network_with_CONTrolZ_flow_0.npy']
    check_output(cap.output, patterns)
