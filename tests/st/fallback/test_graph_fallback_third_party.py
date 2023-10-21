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
import operator
import builtins
import functools
import pytest
import numpy as np
from numpy import logspace
import scipy
from scipy.linalg import qr

import mindspore as ms
from mindspore import nn
from . import utils

ms.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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
