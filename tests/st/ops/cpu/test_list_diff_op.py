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

import pytest
import mindspore as ms
import mindspore.context as context
from mindspore import nn, Tensor
from mindspore.ops.operations import array_ops as ops


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = ops.ListDiff()

    def construct(self, x, y):
        return self.op(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
def test_list_diff_dynamic_shape():
    """
    Feature: test ListDiff op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    input_x_dyn = Tensor(shape=[None], dtype=ms.int32)
    input_y_dyn = Tensor(shape=[None], dtype=ms.int32)
    net.set_inputs(input_x_dyn, input_y_dyn)
    input_x = Tensor([1, 2, 3, 4, 5, 6], dtype=ms.int32)
    input_y = Tensor([1, 3, 5], dtype=ms.int32)
    out, idx = net(input_x, input_y)
    expect_shape = (3,)
    assert out.asnumpy().shape == expect_shape
    assert idx.asnumpy().shape == expect_shape
