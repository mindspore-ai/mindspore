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
""" test graph fallback """
import os
import operator
import builtins
import functools
import subprocess
import pytest
import numpy as np
from numpy import logspace
import scipy
from scipy.linalg import qr

import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import initializer
from . import utils
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_asnumpy():
    """
    Feature: JIT Fallback
    Description: Test tensor.asnumpy().
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return ms.Tensor(x.asnumpy())

    x = ms.Tensor(1)
    assert func(x) == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_relative_import():
    """
    Feature: JIT Fallback
    Description: Test relative imported modules.
    Expectation: No exception.
    """
    @ms.jit
    def func(x, y):
        return utils.add_func(x, y)

    x = ms.Tensor(2, dtype=ms.int32)
    y = ms.Tensor(3, dtype=ms.int32)
    assert func(x, y) == 5


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_operator_add():
    """
    Feature: JIT Fallback
    Description: Test operator.add in graph mode
    Expectation: No exception.
    """
    @ms.jit
    def func(x, y):
        return operator.add(x, y)

    x = ms.Tensor(1, dtype=ms.int32)
    y = ms.Tensor(2, dtype=ms.int32)
    assert func(x, y) == 3


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_builtin_abs():
    """
    Feature: JIT Fallback
    Description: Test builtins.abs in graph mode
    Expectation: No exception.
    """
    @ms.jit
    def func(x):
        return builtins.abs(x)

    x = ms.Tensor(-2, dtype=ms.int32)
    assert func(x) == 2


def add_func(x, y):
    return x + y


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_functools_partial():
    """
    Feature: JIT Fallback
    Description: Test functools.partial in graph mode.
    Expectation: No exception.
    """
    class ModuleNet(nn.Cell):
        def construct(self, x, y):
            func = functools.partial(add_func, x)
            out = func(y)
            return out

    x = ms.Tensor([1, 2, 3], ms.int32)
    y = ms.Tensor([4, 5, 6], ms.int32)
    net = ModuleNet()
    out = net(x, y)
    assert np.all(out.asnumpy() == np.array([5, 7, 9]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_np_logspace_func():
    """
    Feature: JIT Fallback
    Description: Test numpy.logspace in graph mode.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return logspace(0, 0, 5)

    out = func()
    expect = np.array([1., 1., 1., 1., 1.])
    assert np.allclose(out, expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_scipy_concatenate():
    """
    Feature: JIT Fallback
    Description: Test scipy.linalg in graph mode.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        return scipy.concatenate((x, y))

    out = func()
    expect = np.array([1, 2, 3, 4, 5, 6])
    assert np.all(out == expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_scipy_linalg_qr():
    """
    Feature: JIT Fallback
    Description: Test scipy.linalg in graph mode.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        x = [[1, 2], [3, 4]]
        return qr(x)

    out = func()
    assert out[0].shape == (2, 2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_env_ms_jit():
    """
    Feature: JIT Fallback
    Description: Test environ variables in graph mode.
    Expectation: No exception.
    """
    @ms.jit
    def foo():
        t = initializer('ones', [1, 2, 3], ms.float32)
        return t

    os.environ['MS_JIT'] = '0'
    out = foo()
    assert out.shape == (1, 2, 3)
    os.environ['MS_JIT'] = '1'
    with pytest.raises(RuntimeError):
        foo()
    del os.environ['MS_JIT']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_env_ms_jit_modules():
    """
    Feature: JIT Fallback
    Description: Test environ variables in graph mode.
    Expectation: No exception.
    """
    @ms.jit
    def func():
        return logspace(0, 0, 5)

    os.environ['MS_JIT_MODULES'] = 'numpy'
    with pytest.raises(Exception):
        func()
    os.environ['MS_JIT_IGNORE_MODULES'] = 'numpy'
    out = func()
    expect = np.array([1., 1., 1., 1., 1.])
    assert np.allclose(out, expect)
    del os.environ['MS_JIT_MODULES']
    del os.environ['MS_JIT_IGNORE_MODULES']


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_env_fallback_dump_node():
    """
    Feature: JIT Fallback
    Description: Test environ variables in graph mode.
    Expectation: No exception.
    """
    def check_dump_node_log():
        # Clear log files
        log_file_name = "fallback_dump_node.log"
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        assert not os.path.exists(log_file_name)

        cmd_first = f"GLOG_v=3 pytest -sv test_graph_fallback_third_party.py::test_tensor_asnumpy > " + \
            log_file_name + " 2>&1"
        subprocess.check_output(cmd_first, shell=True)
        assert os.path.exists(log_file_name)
        with open(log_file_name, "r") as f_first:
            data_first = f_first.read()

        expected_msg = "Found unsupported syntax in graph mode, those codes would be fallen back to Python interpreter"
        match = expected_msg in data_first

        # Clean files
        os.remove(log_file_name)
        return match

    os.environ['MS_DEV_FALLBACK_DUMP_NODE'] = '0'
    assert not check_dump_node_log()
    os.environ['MS_DEV_FALLBACK_DUMP_NODE'] = '1'
    assert check_dump_node_log()
    del os.environ['MS_DEV_FALLBACK_DUMP_NODE']
