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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.squaresumall = P.SquareSumAll()

    def construct(self, x0, x1):
        return self.squaresumall(x0, x1)


def get_output(inp0, inp1, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(inp0, inp1)
    return output


def run_basic(datatype):
    inp0 = Tensor(np.random.normal(1, 0.1, [800, 96]).astype(datatype))
    inp1 = Tensor(np.random.normal(1, 0.1, [800, 96]).astype(datatype))
    expect = get_output(inp0, inp1, False)
    output = get_output(inp0, inp1, True)
    expect_np0 = expect[0].asnumpy().copy()
    output_np0 = output[0].asnumpy().copy()
    expect_np1 = expect[1].asnumpy().copy()
    output_np1 = output[1].asnumpy().copy()
    assert np.allclose(expect_np0, output_np0, 1.e-4, 1.e-7)
    assert np.allclose(expect_np1, output_np1, 1.e-4, 1.e-7)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_1():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_gpu_2():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic(np.float32)
