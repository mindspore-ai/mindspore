# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class NetElu(nn.Cell):
    def __init__(self):
        super(NetElu, self).__init__()
        self.elu = P.Elu()

    def construct(self, x):
        return self.elu(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_elu_fp16():
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float16))
    expect = np.array([[-0.632, 4.0, -0.999], [2.0, -0.993, 9.0]]).astype(np.float16)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    elu = NetElu()
    output = elu(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    elu = NetElu()
    output = elu(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_elu_fp32():
    x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]).astype(np.float32))
    expect = np.array([[-0.632, 4.0, -0.999], [2.0, -0.993, 9.0]]).astype(np.float32)
    error = np.ones(shape=[2, 3]) * 1.0e-6

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    elu = NetElu()
    output = elu(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    elu = NetElu()
    output = elu(x)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)


def test_elu_functional_api():
    """
    Feature: test elu functional API.
    Description: test elu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mstype.float32)
    output = F.elu(input_x)
    expected = np.array([[-0.63212055, 4.0, -0.99966455], [2.0, -0.99326205, 9.0]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_elu_functional_api_modes():
    """
    Feature: test elu functional API for different modes.
    Description: test elu functional API and compare with expected output.
    Expectation: output should be equal to expected value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_elu_functional_api()
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_elu_functional_api()
