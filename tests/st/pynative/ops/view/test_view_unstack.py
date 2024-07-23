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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_unstack_single_op():
    """
    Feature: unstack
    Description: Verify the result of unstack
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            return ops.unbind(x, dim=y)

    net = Net()
    np_tensor = np.random.randn(10, 1, 1, 20)
    ms_tensor = Tensor(np_tensor, dtype=ms.float32)

    expect_output_tuple = net(ms_tensor, 0)

    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad_tuple = grad_op(net)(ms_tensor, 0)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    output_tuple = net(ms_tensor, 0)
    grad_tuple = grad_op(net)(ms_tensor, 0)
    for expect_output, output in zip(expect_output_tuple, output_tuple):
        np.testing.assert_array_equal(expect_output.asnumpy(), output.asnumpy())

    for expect_grad, grad in zip(expect_grad_tuple, grad_tuple):
        np.testing.assert_allclose(expect_grad[0].asnumpy(), grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_unstack_multiple_op():
    """
    Feature: unstack
    Description: Verify the result of unstack
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            temp = ops.unbind(x, dim=y)
            temp = (temp[0] + 1) * 2
            return ops.unbind(temp, dim=y)

    net = Net()
    np_tensor = np.random.randn(10, 1, 1, 20)
    ms_tensor = Tensor(np_tensor, dtype=ms.float32)

    expect_output_tuple = net(ms_tensor, 0)

    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad_tuple = grad_op(net)(ms_tensor, 0)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    output_tuple = net(ms_tensor, 0)
    grad_tuple = grad_op(net)(ms_tensor, 0)
    for expect_output, output in zip(expect_output_tuple, output_tuple):
        np.testing.assert_array_equal(expect_output.asnumpy(), output.asnumpy())

    for expect_grad, grad in zip(expect_grad_tuple, grad_tuple):
        np.testing.assert_allclose(expect_grad[0].asnumpy(), grad[0].asnumpy(), 0.00001, 0.00001)
