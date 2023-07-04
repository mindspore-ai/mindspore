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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
