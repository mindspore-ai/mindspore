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


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_diagonal_single_op():
    """
    Feature: diagonal
    Description: Verify the result of diagonal
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            return ops.diagonal(x, *y)

    input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_perm = (0, 0, 1)
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_diagonal_multiple_op():
    """
    Feature: diagonal
    Description: Verify the result of diagonal
    Expectation: success
    """
    ms.set_context(mode=ms.GRAPH_MODE)

    class Net(nn.Cell):
        def construct(self, x, y):
            temp = ops.diagonal(x, *y)
            temp = (temp + 1) * 2
            return ops.diagonal(temp, *y)

    input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    input_perm = (0, 0, 1)
    net = Net()
    expect_output = net(input_x, input_perm).asnumpy()
    grad_op = ops.GradOperation(get_all=True, get_by_list=False, sens_param=False)
    expect_grad = grad_op(net)(input_x, input_perm)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    net = Net()
    output = net(input_x, input_perm).asnumpy()
    grad = grad_op(net)(input_x, input_perm)
    np.testing.assert_array_equal(output, expect_output)
    np.testing.assert_allclose(grad[0].asnumpy(), expect_grad[0].asnumpy(), 0.00001, 0.00001)
