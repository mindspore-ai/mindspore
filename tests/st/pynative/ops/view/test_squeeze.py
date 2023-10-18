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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_squeeze_single_op():
    """
    Feature: squeeze
    Description: Verify the result of squeeze
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y=None):
            return ops.squeeze(x, axis=y)

    net = Net()
    np_tensor = np.random.randn(10, 1, 1, 20)
    ms_tensor = Tensor(np_tensor, dtype=ms.float32)
    # 测试axis为空
    expect_output_1 = net(ms_tensor).asnumpy()
    # 将输入axis的维度为1的维度删除
    expect_output_2 = net(ms_tensor, 1).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad_1 = grad_op(net)(ms_tensor)
    expect_grad_2 = grad_op(net)(ms_tensor, 1)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    output_1 = net(ms_tensor).asnumpy()
    output_2 = net(ms_tensor, 1).asnumpy()
    grad_1 = grad_op(net)(ms_tensor)
    grad_2 = grad_op(net)(ms_tensor, 1)
    np.testing.assert_array_equal(output_1, expect_output_1)
    np.testing.assert_array_equal(output_2, expect_output_2)
    assert np.allclose(grad_1[0].asnumpy(), expect_grad_1[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(grad_2[0].asnumpy(), expect_grad_2[0].asnumpy(), 0.00001, 0.00001)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_squeeze_multiple_op():
    """
    Feature: squeeze
    Description: Verify the result of squeeze
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y=None):
            temp = ops.squeeze(x, y)
            temp = (temp + 1) * 2
            return ops.squeeze(temp, y)

    net = Net()
    np_tensor = np.random.randn(10, 1, 1, 20)
    ms_tensor = Tensor(np_tensor, dtype=ms.float32)

    expect_output = net(ms_tensor, 1).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(ms_tensor, 1)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    output = net(ms_tensor, 1).asnumpy()
    grad = grad_op(net)(ms_tensor, 1)
    np.testing.assert_array_equal(output, expect_output)
    assert np.allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)
