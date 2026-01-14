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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.addcmul = P.Addcmul()

    def construct(self, input_data, x1, x2, value):
        return self.addcmul(input_data, x1, x2, value)


def get_output(input_data, x1, x2, value, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net = Net()
    output = net(input_data, x1, x2, value)
    return output


def run_basic(shape1, shape2, shape3, shape4, dtype):
    np.random.seed(0)
    input_data = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    x1 = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    x2 = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    value = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    expect = get_output(input_data, x1, x2, value, False)
    output = get_output(input_data, x1, x2, value, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_basic_ascend_f16():
    """
    Feature: test graph kernel Addcmul
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level="O0")
    context.set_context(mode=context.GRAPH_MODE)
    run_basic([3], [1, 3], [3, 1], [1], np.float16)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_basic_ascend_f32():
    """
    Feature: test graph kernel Addcmul
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level="O0")
    context.set_context(mode=context.GRAPH_MODE)
    run_basic([3], [6, 3], [6, 3], [6, 3], np.float32)
