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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
from mindspore import ops, Tensor
import mindspore.common.dtype as mstype


def irfft_forward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    x = ops.FFTWithSize(signal_ndim, inverse, real, norm, onesided, signal_sizes)(x)
    return ops.FFTWithSize(signal_ndim, not inverse, real, norm, onesided, signal_sizes)(x)

def irfft_backward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    return ops.grad(irfft_forward_func, (0,))(x, signal_ndim, inverse, real, norm, onesided, signal_sizes)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_irfft(mode):
    """
    Feature: irfft function
    Description: test cases for irfft
    Expectation: The result match to the expect value
    """
    ms.context.set_context(mode=mode)
    x_dtype = [np.float32, mstype.complex128]
    x_shape = [10, 3, 3]
    x_real = np.random.randn(*x_shape).astype(x_dtype[0])
    x_img = np.random.randn(*x_shape).astype(x_dtype[0])
    x = Tensor(x_real + 1j * x_img, dtype=x_dtype[1])
    output = irfft_backward_func(x, 1, True, True, signal_sizes=(4,))
    assert output.shape == (10, 3, 3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_irfft_signal_sizes(mode):
    """
    Feature: irfft function
    Description: test cases for irfft when signal_sizes=()
    Expectation: The result match to the expect value
    """
    ms.context.set_context(mode=mode)
    x_dtype = [np.float32, mstype.complex128]
    x_shape = [10, 3, 3]
    x_real = np.random.randn(*x_shape).astype(x_dtype[0])
    x_img = np.random.randn(*x_shape).astype(x_dtype[0])
    x = Tensor(x_real + 1j * x_img, dtype=x_dtype[1])
    output = irfft_forward_func(x, 1, True, True, 'forward')
    assert output.shape == (10, 3, 3)
