# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
""" test graph fallback """
import os
import sys
import time
import tempfile
from contextlib import contextmanager
import pytest
import numpy as np
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, jit, context
from tests.security_utils import security_off_wrap

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


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
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_np_print_1():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def np_print():
        x = np.array([1, 2, 3, 4, 5])
        print("x: ", x)
        return Tensor(x)

    cap = Capture()
    with capture(cap):
        res = np_print()
        assert np.all(res.asnumpy() == np.array([1, 2, 3, 4, 5]))
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'x:  [1 2 3 4 5]'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_np_print_2():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    class PrintNet(nn.Cell):
        def construct(self):
            x = np.array([1, 2, 3, 4, 5])
            print("x: ", x)
            return Tensor(x)

    cap = Capture()
    with capture(cap):
        net = PrintNet()
        res = net()
        assert np.all(res.asnumpy() == np.array([1, 2, 3, 4, 5]))
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'x:  [1 2 3 4 5]'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_tensor_print_1():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def np_print():
        x = np.array([1, 2, 3, 4, 5])
        print("Tensor(x): ", Tensor(x))
        return Tensor(x)

    cap = Capture()
    with capture(cap):
        res = np_print()
        assert np.all(res.asnumpy() == np.array([1, 2, 3, 4, 5]))
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'Tensor(x): \nTensor(shape=[5], dtype=Int64, value=[1 2 3 4 5])\n\n'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_cnode_1():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func(x, y):
        res_sum = x + y
        print("res_sum: ", res_sum)
        return res_sum

    cap = Capture()
    with capture(cap):
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        res = print_func(x, y)
        assert (res.asnumpy() == [2, 4, 6, 8, 10]).all()
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'res_sum: \nTensor(shape=[5], dtype=Int64, value=[ 2  4  6  8 10])\n'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_cnode_2():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        res_sum = x + y
        print("res_sum: ", res_sum)
        return res_sum

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert (res.asnumpy() == [2, 4, 6, 8, 10]).all()
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'res_sum: \nTensor(shape=[5], dtype=Int64, value=[ 2  4  6  8 10])\n'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_cnode_3():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func():
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1, 2, 3, 4, 5])
        res_sum = x + y
        print("res_sum: ", res_sum)
        return Tensor(res_sum)

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert (res.asnumpy() == [2, 4, 6, 8, 10]).all()
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'res_sum:  [ 2  4  6  8 10]'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_validate_tuple():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        tensor_sum = x + y
        print("tensor_sum: ", tensor_sum)
        np_x = np.array([1, 2, 3, 4, 5])
        np_y = np.array([1, 2, 3, 4, 5])
        np_sum = np_x + np_y
        print("np_sum: ", np_sum)
        return tensor_sum, np_sum

    with pytest.raises(RuntimeError) as err:
        res1, res2 = print_func()
        print("res1: ", res1)
        print("res2: ", res2)
    assert "Should not use Python object in runtime" in str(err.value)


@security_off_wrap
@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_validate():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func():
        np_x = np.array([1, 2, 3, 4, 5])
        np_y = np.array([1, 2, 3, 4, 5])
        np_sum = np_x + np_y
        print("np_sum: ", np_sum)
        return np_sum

    res = print_func()
    assert (res == [2, 4, 6, 8, 10]).all()


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_format_np():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func():
        np_x = np.array([1, 2, 3, 4, 5])
        np_y = np.array([1, 2, 3, 4, 5])
        np_sum = np_x + np_y
        print("np_sum: {}".format(np_sum))
        return Tensor(np_sum)

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert (res.asnumpy() == [2, 4, 6, 8, 10]).all()
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'np_sum: [ 2  4  6  8 10]'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_format_tensor():
    """
    Feature: JIT Fallback
    Description: Support print.
    Expectation: No exception.
    """
    @jit
    def print_func():
        x = Tensor(np.array([1, 2, 3, 4, 5]))
        y = Tensor(np.array([1, 2, 3, 4, 5]))
        tensor_sum = x + y
        print("tensor_sum: {}".format(tensor_sum))
        return tensor_sum

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert (res.asnumpy() == [2, 4, 6, 8, 10]).all()
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {'tensor_sum: Tensor(shape=[5], dtype=Int64, value=[ 2  4  6  8 10])\n'}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_string_format():
    """
    Feature: JIT Fallback
    Description: Support print(string % var).
    Expectation: No exception.
    """
    @jit
    def print_func():
        print("I'm %s. I'm %d years old." % ('MindSpore', 3))
        return 0

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert res == 0
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {"I'm MindSpore. I'm 3 years old.\n"}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_string_add_string():
    """
    Feature: JIT Fallback
    Description: Support print(string + string).
    Expectation: No exception.
    """
    def name():
        return "MindSpore"

    @jit
    def print_func():
        print("I'm " + name() + ". I'm 3 years old.")
        return 0

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert res == 0
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {"I'm MindSpore. I'm 3 years old.\n"}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_list():
    """
    Feature: JIT Fallback
    Description: Support print(list).
    Expectation: No exception.
    """
    @jit
    def print_func():
        list_x = [1, 2, 3, 4, 5]
        print("list_x:", list_x)
        return 0

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert res == 0
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {"list_x:\nTensor(shape=[5], dtype=Int64, value=[1 2 3 4 5])\n"}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_tuple():
    """
    Feature: JIT Fallback
    Description: Support print(tuple).
    Expectation: No exception.
    """
    @jit
    def print_func():
        tuple_x = (6, 7, 8, 9, 10)
        print("tuple_x:", tuple_x)
        return 0

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert res == 0
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {"tuple_x:\nTensor(shape=[5], dtype=Int64, value=[ 6  7  8  9 10])\n"}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_dict():
    """
    Feature: JIT Fallback
    Description: Support print(dict).
    Expectation: No exception.
    """
    @jit
    def print_func():
        dict_x1 = dict(zip(['one', 'two', 'three'], [1, 2, 3]))
        dict_x2 = dict([("one", 1), ("two", 2)])
        print("dict_x1:", dict_x1)
        print("dict_x2:", dict_x2)
        return 0

    cap = Capture()
    with capture(cap):
        res = print_func()
        assert res == 0
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {"dict_x1: {'one': 1, 'two': 2, 'three': 3}\n"
                "dict_x2: {'one': 1, 'two': 2}\n"}
    check_output(cap.output, patterns)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_print_exception():
    """
    Feature: graph print.
    Description: Test print.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, input_x, input_y):
            tensor_sum = input_x + input_y
            x = np.array([1, 2, 3, 4, 5])
            y = np.array([1, 2, 3, 4, 5])
            np_sum = x + y
            print("np_sum: ", np_sum, "tensor_sum: ", tensor_sum)
            return tensor_sum, ms.Tensor(np_sum)

    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.array([1, 2, 3, 4, 5]))
    y = ms.Tensor(np.array([1, 2, 3, 4, 5]))
    net = Net()
    net(x, y)


@security_off_wrap
@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_print_joinedstr():
    """
    Feature: graph print joinedstr.
    Description: Test print joinedstr.
    Expectation: No exception.
    """
    @jit
    def np_print():
        x = (1, 2, 3, 4, 5)
        c = f"x:{x}"
        dict_input = {"a": 1, "b": 2, c: 3}
        print(f"Tensor(x): {Tensor(x)}, dict_input: {dict_input}")
        return Tensor(x)

    cap = Capture()
    with capture(cap):
        res = np_print()
        assert np.all(res.asnumpy() == np.array([1, 2, 3, 4, 5]))
        sys.stdout.flush()
        time.sleep(0.1)

    patterns = {"Tensor(x): [1 2 3 4 5], dict_input: {'a': 1, 'b': 2, 'x:(1, 2, 3, 4, 5)': 3}"}
    check_output(cap.output, patterns)
