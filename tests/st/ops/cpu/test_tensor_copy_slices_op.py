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

import mindspore as ms
import mindspore.ops.operations._inner_ops as P
from mindspore import nn, Tensor, context


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = P.TensorCopySlices()

    def construct(self, x, value, begin, end, strides):
        return self.op(x, value, begin, end, strides)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor_copy_slices_dyn():
    """
    Feature: Test TensorCopySlices ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()

    x_dyn = Tensor(shape=[None, None], dtype=ms.int32)
    values_dyn = Tensor(shape=[None, None], dtype=ms.int32)
    begin = (3, 0)
    end = (5, 5)
    strides = (1, 1)
    net.set_inputs(x_dyn, values_dyn, begin, end, strides)

    x = Tensor(np.zeros((5, 5)), dtype=ms.int32)
    values = Tensor(np.ones((2, 5)), dtype=ms.int32)
    out = net(x, values, begin, end, strides)

    expect_shape = (5, 5)
    assert out.asnumpy().shape == expect_shape
