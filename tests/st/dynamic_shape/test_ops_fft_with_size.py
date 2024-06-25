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

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def fft_forward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    return ops.FFTWithSize(signal_ndim, inverse, real, norm, onesided, signal_sizes)(x)


@test_utils.run_with_cell
def fft_backward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    return ops.grad(fft_forward_func, (0,))(x, signal_ndim, inverse, real, norm, onesided, signal_sizes)


pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_forward(mode):
    """
    Feature: auto ops.
    Description: test op fft_with_size.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(2 * 3).reshape(2, 3), ms.complex64)
    output = fft_forward_func(x, 2, True, True)
    expect = np.array([[2.5, -0.5, 0., -0.5], [-1.5, 0., 0., 0.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_backward(mode):
    """
    Feature: auto grad.
    Description: test auto grad of op fft_with_size.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(2 * 3).reshape(2, 3), ms.float32)
    grads = fft_backward_func(x, 2, False, True)
    expect = np.array([[4., 1., 1.], [0., 0., 0.]], dtype=np.float32)
    assert np.allclose(grads.asnumpy(), expect)
