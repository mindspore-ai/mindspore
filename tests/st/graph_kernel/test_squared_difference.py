# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.squared_difference = P.SquaredDifference()

    def construct(self, x, y):
        return self.squared_difference(x, y)


def get_output(x, y, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(x, y)
    return output


def run_squared_difference(shape1, shape2, dtype):
    x = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    y = Tensor(np.random.normal(0, 10, shape2).astype(dtype))
    expect = get_output(x, y, False)
    output = get_output(x, y, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_squared_difference_gpu():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_squared_difference((4, 3), (4, 3), np.float16)
    run_squared_difference((6, 2), (1), np.int32)
    run_squared_difference((1), (4, 3), np.float32)
