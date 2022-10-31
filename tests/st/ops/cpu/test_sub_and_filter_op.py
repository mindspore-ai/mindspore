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

import pytest

import mindspore as ms
import mindspore.ops.operations._embedding_cache_ops as P
from mindspore import nn, Tensor, context


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = P.SubAndFilter()

    def construct(self, x, max_num, offset):
        return self.op(x, max_num, offset)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sub_and_filter_dyn():
    """
    Feature: Test SubAndFilter ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()

    x_dyn = Tensor(shape=[None], dtype=ms.int32)
    max_num = 10
    offset = 5
    net.set_inputs(x_dyn, max_num, offset)

    x = Tensor([1, 3, 5, 8, 9, 16], dtype=ms.int32)
    out = net(x, max_num, offset)

    expect_shapes = [(3,), (3,)]
    for i in range(2):
        assert out[i].asnumpy().shape == expect_shapes[i]
