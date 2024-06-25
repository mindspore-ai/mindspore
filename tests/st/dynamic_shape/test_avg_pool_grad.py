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
from mindspore import Tensor
import mindspore.context as context
import mindspore.nn as nn
from mindspore import ops
from mindspore.ops import composite as C
from tests.mark_utils import arg_mark


class AvgPool(nn.Cell):
    def __init__(self, kernel_size, strides, pad_mode, data_format):
        super(AvgPool, self).__init__()
        self.avgpool = ops.AvgPool(kernel_size=kernel_size, strides=strides, pad_mode=pad_mode, data_format=data_format)

    def construct(self, x):
        return self.avgpool(x)


class AvgPoolGrad(nn.Cell):
    def __init__(self, forward):
        super(AvgPoolGrad, self).__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)

    def construct(self, x, sens):
        return self.grad(self.forward)(x, sens)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avg_pool_grad(mode):
    """
    Feature: test avg_pool_grad op.
    Description: backward.
    Expectation: expect correct forward and backward result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[[10, 1, 2, 3, -4, -5],
                           [6, 7, 8, 9, -10, -11],
                           [12, 13, 24, -15, -16, -17],
                           [18, 19, 20, 21, 22, 23],
                           [32, 25, 26, 27, 28, 40],
                           [30, 31, 35, 33, 34, 35]]]]).astype(np.float32))
    avgpool = AvgPool(kernel_size=2, strides=2, pad_mode="VALID", data_format="NCHW")
    actual_output = avgpool(x)
    avgpool_grad = AvgPoolGrad(avgpool)
    sens = Tensor(np.arange(1, 10).reshape(
        actual_output.shape).astype(np.float32))
    actual_grad = avgpool_grad(x, sens)
    expect_grad = np.array([[[[0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
                              [0.25, 0.25, 0.5, 0.5, 0.75, 0.75],
                              [1., 1., 1.25, 1.25, 1.5, 1.5],
                              [1., 1., 1.25, 1.25, 1.5, 1.5],
                              [1.75, 1.75, 2., 2., 2.25, 2.25],
                              [1.75, 1.75, 2., 2., 2.25, 2.25]]]]).astype(np.float32)
    assert (actual_grad[0].asnumpy() == expect_grad).all()
