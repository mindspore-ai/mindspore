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
def test_transpose_single_op():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            return ops.transpose(x, y)

    input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_perm = (0, 2, 1)
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    assert np.allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_transpose_multiple_op():
    """
    Feature: transpose
    Description: Verify the result of transpose
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            temp = ops.transpose(x, y)
            temp = (temp + 1) * 2
            return ops.transpose(temp, y)

    input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_perm = (0, 2, 1)
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    assert np.allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)
