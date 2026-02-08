# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore import Tensor, mint
from mindspore.nn import Cell


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bmm = mint.bmm

    def construct(self, x, y):
        return self.bmm(x, y)


def get_output(x, y, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(jit_level='O1')
    else:
        context.set_context(jit_level='O0')
    net = Net()
    output = net(x, y)
    return output


def run_basic(shape1, shape2, dtype):
    x = Tensor(np.random.normal(0, 1, shape1), dtype)
    y = Tensor(np.random.normal(0, 1, shape2), dtype)
    expect = get_output(x, y, False)
    output = get_output(x, y, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_basic_ascend_f16():
    """
    Feature: test graph kernel mint.bmm
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic((4, 256, 128), (4, 128, 256), mindspore.float16)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=6.0)
def test_basic_ascend_f32():
    """
    Feature: test graph kernel mint.bmm
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_basic((4, 256, 128), (4, 128, 256), mindspore.float32)
