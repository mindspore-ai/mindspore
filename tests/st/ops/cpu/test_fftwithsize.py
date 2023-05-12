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

import numpy as np
import pytest

import mindspore.context as context
from mindspore import Tensor
import mindspore.ops as F


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.fft(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)

    output_ifft = F.ifft(output)
    diff_ifft = np.abs(output_ifft.asnumpy() - x.asnumpy())
    assert np.all(diff_ifft < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.fft2(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)

    output_ifft2 = F.ifft2(output)
    diff_ifft2 = np.abs(output_ifft2.asnumpy() - x.asnumpy())
    assert np.all(diff_ifft2 < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.complex64, 1e-6), (np.complex128, 1e-6)])
def test_fftwithsize_fftn_ifftn(dtype, eps):
    """
    Feature: fftn & ifftn function
    Description: test cases for fftn & ifftn
    Expectation: the result matches pytorch
    """
    x = Tensor(np.array([[[1.6243454 +0.j, -0.6117564 +0.j, -0.5281718 +0.j],
                          [-1.0729686 +0.j, 0.86540765+0.j, -2.3015387 +0.j]]]).astype(dtype))
    expect = np.array([[[-2.02468245+0.j, 1.83940642-2.6702696j, 1.83940642+2.6702696j],
                        [2.99351685+0.j, 2.54921257+2.81504238j, 2.54921257-2.81504238j]]]).astype(dtype)
    error = np.ones(shape=[1, 2, 3]) * eps
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    output = F.fftn(x)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(diff < error)

    output_ifftn = F.ifftn(output)
    diff_ifftn = np.abs(output_ifftn.asnumpy() - x.asnumpy())
    assert np.all(diff_ifftn < error)
