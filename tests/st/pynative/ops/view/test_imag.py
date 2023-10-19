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
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_imag_single_op():
    """
    Feature: imag
    Description: Verify the result of imag
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            return ops.imag(x)

    a = np.random.randn(3, 4, 5)
    conj = a + 1j * np.random.randn(3, 4, 5)
    input_x = Tensor(conj, ms.complex64)
    net = Net()
    expect_output = net(input_x).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x).asnumpy()
    grad = grad_op(net)(input_x)
    np.testing.assert_array_equal(output, expect_output)
    assert np.allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_imag_multiple_op():
    """
    Feature: imag
    Description: Verify the result of imag
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x):
            temp = ops.imag(x)
            temp = (temp + 1) * 2
            return ops.imag(temp)

    a = np.random.randn(3, 4, 5)
    conj = a + 1j * np.random.randn(3, 4, 5)
    input_x = Tensor(conj, ms.complex64)
    net = Net()
    expect_output = net(input_x).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x).asnumpy()
    grad = grad_op(net)(input_x)
    np.testing.assert_array_equal(output, expect_output)
    assert np.allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)
