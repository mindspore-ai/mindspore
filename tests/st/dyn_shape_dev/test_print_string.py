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
import time
import tempfile
from contextlib import contextmanager
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, jit
from mindspore.ops import operations as P
from tests.security_utils import security_off_wrap
import test_utils


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


@security_off_wrap
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_func
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
