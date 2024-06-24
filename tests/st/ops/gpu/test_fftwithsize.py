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

import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor, ops
from tests.st.utils import test_utils

@test_utils.run_with_cell
def fft_forward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    return ops.FFTWithSize(signal_ndim, inverse, real, norm, onesided, signal_sizes)(x)

@test_utils.run_with_cell
def rfft_and_irfft_forward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    x = ops.FFTWithSize(signal_ndim, inverse, real, norm, onesided, signal_sizes)(x)
    return ops.FFTWithSize(signal_ndim, not inverse, real, norm, onesided, signal_sizes)(x)

@test_utils.run_with_cell
def fft_backward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    return ops.grad(fft_forward_func, (0,))(x, signal_ndim, inverse, real, norm, onesided, signal_sizes)

@test_utils.run_with_cell
def rfft_and_irfft_backward_func(x, signal_ndim, inverse, real, norm='backward', onesided=True, signal_sizes=()):
    return ops.grad(rfft_and_irfft_forward_func, (0,))(x, signal_ndim, inverse, real, norm, onesided, signal_sizes)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype, eps', [(np.complex64, 1e-6), (np.complex128, 1e-6)])
def test_fftwithsize_fft_ifft(dtype, eps):
    """
    Feature: fft & ifft function
    Description: test cases for fft & ifft
    Expectation: the result matches pytorch
    """
    x = Tensor(np.array([1.6243454+0.j, -0.6117564+0.j, -0.5281718+0.j, -1.0729686+0.j]).astype(dtype))
    expect = np.array([-0.5885514+0.j, 2.1525173-0.46121222j, 2.7808986+0.j, 2.1525173+0.46121222j]).astype(dtype)
    error = np.ones(shape=[4]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    output = fft_forward_func(x, 1, False, False)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)

    output_ifft = fft_forward_func(output, 1, True, False)
    diff_ifft = np.abs(output_ifft.asnumpy() - x.asnumpy())
    assert np.all(diff_ifft < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('dtype, eps', [(np.complex64, 1e-6), (np.complex128, 1e-6)])
def test_fftwithsize_fft2_ifft2(dtype, eps):
    """
    Feature: fft2 & ifft2 function
    Description: test cases for fft2 & ifft2
    Expectation: the result matches pytorch
    """
    x = Tensor(np.array([[1.6243454+0.j, -0.6117564+0.j], [-0.5281718+0.j, -1.0729686+0.j]]).astype(dtype))
    expect = np.array([[-0.5885514+0.j, 2.7808986+0.j], [2.6137295+0.j, 1.6913052+0.j]]).astype(dtype)
    error = np.ones(shape=[2, 2]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    output = fft_forward_func(x, 2, False, False)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)

    output_ifft2 = fft_forward_func(output, 2, True, False)
    diff_ifft2 = np.abs(output_ifft2.asnumpy() - x.asnumpy())
    assert np.all(diff_ifft2 < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_rfft3_forward(mode):
    """
    Feature: rfft3 forward function
    Description: test cases for rfft
    Expectation: the result matches pytorch
    """
    ms.context.set_context(mode=mode)
    x = np.arange(1 * 2 * 3 * 4, dtype=np.float64).reshape(1, 2, 3, 4)
    ms_x = ms.Tensor(x)
    output = fft_forward_func(ms_x, 3, False, True)
    expect = np.fft.rfftn(x, s=(2, 3, 4))
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_irfft3_forward(mode):
    """
    Feature: irfft3 forward function
    Description: test cases for irfft3
    Expectation: the result matches pytorch
    """
    ms.context.set_context(mode=mode)
    x = np.arange(1 * 2 * 3 * 3, dtype=np.complex128).reshape(1, 2, 3, 3)
    ms_x = ms.Tensor(x)
    output = fft_forward_func(ms_x, 3, True, True)
    expect = np.fft.irfftn(x, s=(2, 3, 4))
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_rfft3_backward(mode):
    """
    Feature: rfft3 backward function
    Description: test cases for rfft3
    Expectation: the result matches pytorch
    """
    ms.context.set_context(mode=mode)
    dim1 = 1
    dim2 = 2
    dim3 = 3
    dim4 = 4
    offset_size = dim1 * dim2 * dim3 * dim4
    x = np.arange(offset_size, dtype=np.float64).reshape(dim1, dim2, dim3, dim4)
    ms_x = ms.Tensor(x)
    output = fft_backward_func(ms_x, 3, False, True)
    dout = np.ones((dim1, dim2, dim3, dim4 // 2 + 1), dtype=np.complex128)
    concat_array = np.zeros((dim1, dim2, dim3, dim4 - dim4 // 2 - 1))
    concat_array = concat_array.astype(np.complex128)
    dout = np.concatenate((dout, concat_array), axis=-1)
    expect = np.fft.ifftn(dout, s=(dim2, dim3, dim4)) * offset_size
    assert np.allclose(output.asnumpy(), expect.real)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fft_with_size_rfft3_and_irfft3_backward(mode):
    """
    Feature: rfft3_and_irfft3 function
    Description: test cases for rfft3_and_irfft3
    Expectation: the result matches pytorch
    """
    ms.context.set_context(mode=mode)
    dim1 = 1
    dim2 = 2
    dim3 = 3
    dim4 = 4
    offset_size = dim1 * dim2 * dim3 * dim4
    x = np.arange(offset_size, dtype=np.float64).reshape(dim1, dim2, dim3, dim4)
    ms_x = ms.Tensor(x)
    output = rfft_and_irfft_backward_func(ms_x, 3, False, True)
    expect = np.ones((dim1, dim2, dim3, dim4))
    assert np.allclose(output.asnumpy(), expect)
