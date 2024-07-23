# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import functional as F
import mindspore.context as context


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_pdist_normal(dtype, eps):
    """
    Feature: Pdist cpu kernel
    Description: test the Pdist p = 2.0.
    Expectation: the output matches numpy
    """
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=dtype))
    error = np.ones(shape=(3,)) * eps
    output = F.pdist(x, p=2.0)
    expect = np.array([1.41421356, 2.82842712, 1.41421356], dtype=dtype)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_pdist_zero(dtype, eps):
    """
    Feature: Pdist cpu kernel
    Description: test the Pdist p = 0.0.
    Expectation: the output matches numpy
    """
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=dtype))
    error = np.ones(shape=(3,)) * eps
    output = F.pdist(x, p=0.0)
    expect = np.array([2., 2., 2.], dtype=dtype)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('dtype, eps', [(np.float16, 1.0e-3), (np.float32, 1.0e-6), (np.float64, 1.0e-6)])
def test_pdist_inf(dtype, eps):
    """
    Feature: Pdist cpu kernel
    Description: test the Pdist p = inf.
    Expectation: the output matches numpy
    """
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]], dtype=dtype))
    error = np.ones(shape=(3,)) * eps
    output = F.pdist(x, p=float('inf'))
    expect = np.array([1., 2., 1.], dtype=dtype)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)
