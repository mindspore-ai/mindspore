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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P


def cumulative_logsumexp(nptype):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x0 = np.random.rand(2, 3, 4, 4).astype(nptype)
    axis0 = np.array(1).astype(np.int32)

    x1 = np.random.rand(2, 3, 4, 4).astype(nptype)
    axis1 = np.array(3).astype(np.int32)

    x2 = np.random.rand(2, 3, 1, 4).astype(nptype)
    axis2 = np.array(2).astype(np.int32)

    class CumulativeLogsumexp(nn.Cell):
        def __init__(self, nptype):
            super(CumulativeLogsumexp, self).__init__()

            self.x0 = Tensor(x0)
            self.axis0 = Tensor(axis0)

            self.x1 = Tensor(x1)
            self.axis1 = Tensor(axis1)

            self.x2 = Tensor(x2)
            self.axis2 = Tensor(axis2)


        @ms_function
        def construct(self):
            return (P.CumulativeLogsumexp()(self.x0, self.axis0),
                    P.CumulativeLogsumexp()(self.x1, self.axis1),
                    P.CumulativeLogsumexp()(self.x2, self.axis2))

    cumlogsumexp = CumulativeLogsumexp(nptype)
    output = cumlogsumexp()

    expect0 = np.log(np.cumsum(np.exp(x0.astype(np.float64)), axis=axis0)).astype(nptype)
    diff0 = abs(output[0].asnumpy() - expect0)
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output[0].shape == expect0.shape

    expect1 = np.log(np.cumsum(np.exp(x1.astype(np.float64)), axis=axis1)).astype(nptype)
    diff1 = abs(output[1].asnumpy() - expect1)
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output[1].shape == expect1.shape

    expect2 = np.log(np.cumsum(np.exp(x2.astype(np.float64)), axis=axis2)).astype(nptype)
    diff2 = abs(output[2].asnumpy() - expect2)
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output[2].shape == expect2.shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cumulative_logsumexp_float16():
    """
    Feature: Test CumulativeLogsumexp.
    Description: The input type is float16 and the output type is float16.
    Expectation: Check it by expected_output variable.
    """
    cumulative_logsumexp(np.float16)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cumulative_logsumexp_float32():
    """
    Feature: Test CumulativeLogsumexp.
    Description: The input type is float32 and the output type is float32.
    Expectation: Check it by expected_output variable.
    """
    cumulative_logsumexp(np.float32)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cumulative_logsumexp_float64():
    """
    Feature: Test CumulativeLogsumexp.
    Description: The input type is float64 and the output type is float64.
    Expectation: Check it by expected_output variable.
    """
    cumulative_logsumexp(np.float64)
