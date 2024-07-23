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
import os
import sys
import time
import tempfile
from contextlib import contextmanager
import pytest
import numpy as np
from mindspore import Tensor, jit, context, nn, mutable
import mindspore.ops as ops
from tests.mark_utils import arg_mark


context.set_context(mode=context.GRAPH_MODE)


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
    for pattern in patterns:
        assert output.find(pattern) != -1, "Unexpected output:\n" + output + "\n--- pattern ---\n" + pattern


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_print_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test print in fallback runtime
    Expectation: No exception.
    """
    @jit
    def foo():
        x = Tensor(np.array([1, 2, 3, 4]))
        y = x.asnumpy()
        print(x, y)

    cap = Capture()
    with capture(cap):
        res = foo()
        assert not res
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'[1 2 3 4] [1 2 3 4]'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_print_asnumpy_custom_class():
    """
    Feature: JIT Fallback
    Description: Test print in fallback runtime
    Expectation: No exception.
    """
    class GetattrClass():
        def __init__(self):
            self.attr1 = Tensor(np.array([1, 2, 3, 4])).asnumpy()
            self.attr2 = 1

    class GetattrClassNet(nn.Cell):
        def __init__(self):
            super(GetattrClassNet, self).__init__()
            self.cls = GetattrClass()

        def construct(self):
            print(self.cls.attr1)
            print(self.cls.attr2)

    cap = Capture()
    with capture(cap):
        net = GetattrClassNet()
        res = net()
        assert not res
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'[1 2 3 4]\n1\n'}
    check_output(cap.output, patterns)


@pytest.mark.skip(reason="print mutalble is not supported")
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_print_mutable():
    """
    Feature: JIT Fallback
    Description: Test print in fallback runtime
    Expectation: No exception.
    """

    class GetattrClassMutableNet(nn.Cell):
        def construct(self, x):
            print(x)

    cap = Capture()
    with capture(cap):
        net = GetattrClassMutableNet()
        input_x = mutable(2)
        res = net(input_x)
        assert not res
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'2\n'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fallback_print_pyinterpret():
    """
    Feature: JIT Fallback
    Description: Test print in fallback runtime
    Expectation: No exception.
    """
    @jit
    def test_print():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        tensor_sum = x + y
        print("tensor_sum: ", tensor_sum)
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        np_sum = x + y
        print("np_sum: ", np_sum)
        return tensor_sum, Tensor(np_sum)

    cap = Capture()
    with capture(cap):
        out = test_print()
        assert (out[0].asnumpy() == out[1]).all()
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'tensor_sum: \nTensor(shape=[5], dtype=Int64, value=[ 2  4  6  8 10])\nnp_sum:  [ 2  4  6  8 10]\n'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_np_init():
    """
    Feature: JIT Fallback
    Description: Test numpy defined in init in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.x = np.array([1, 2])

        def construct(self):
            y = np.array([3, 4])
            print(self.x)
            return Tensor(y + self.x)

    cap = Capture()
    with capture(cap):
        net = Net()
        res = net()
        assert (res.asnumpy() == [4, 6]).all()
        sys.stdout.flush()
        time.sleep(0.1)
    patterns = {'[1 2]\n'}
    check_output(cap.output, patterns)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_print_str_format():
    """
    Feature: JIT Fallback
    Description: Test print str format in graph mode.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, x):
            x = ops.add(x, x)
            print('x is {}'.format(x))
            x = x - 1
            return x

    cap = Capture()
    with capture(cap):
        input_x = Tensor([[1, 2], [3, 4]])
        net = Net()
        res = net(input_x)
        assert (res.asnumpy() == [[1, 3], [5, 7]]).all()
        sys.stdout.flush()
        time.sleep(0.1)
    patterns = {'x is [[2 4]\n [6 8]]\n'}
    check_output(cap.output, patterns)
