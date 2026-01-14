# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
import mindspore.ops.operations as P


class NetFloatStatusAddN(Cell):
    def __init__(self):
        super(NetFloatStatusAddN, self).__init__()
        self.status = P.FloatStatus()
        self.addn = P.AddN()
        self.square = P.Square()

    def construct(self, x, y, z):
        res0 = self.square(x)
        res1 = self.status(res0)
        res2 = self.status(y)
        res3 = self.status(z)
        res4 = self.addn((res1, res2, res3))
        return self.square(res4)


def run_floatstatus_addn():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    input_z = np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32)
    net = NetFloatStatusAddN()
    result = net(Tensor(input_x), Tensor(input_y), Tensor(input_z))
    res = np.allclose(0, result.asnumpy(), rtol=1.e-4, atol=1.e-7)
    assert res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_floatstatus_addn():
    """
    Feature: graph kernel testcase for floatstatus addn fusion
    Description: random input when using graph_kernel in graph mode
    Expectation: the result is 0
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_floatstatus_addn()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=36.0)
@pytest.mark.parametrize("input_type", [mindspore.float16, mindspore.bfloat16])
def test_floatstatus_addn_ascend(input_type):
    """
    Feature: graph kernel testcase for floatstatus addn fusion
    Description: random input when using graph_kernel in graph mode
    Expectation: the result match with the expected result
    """

    class NetFloatStatus(Cell):
        def __init__(self, shape):
            super(NetFloatStatus, self).__init__()
            self.axis = tuple(range(len(shape)))
            self.one = Tensor(1, mindspore.float32)

        def construct(self, x0):
            y0 = ops.isfinite(x0)
            y1 = ops.ReduceAll(False)(y0, self.axis)
            y2 = ops.cast(y1, mindspore.float32)
            y3 = ops.sub(self.one, y2)
            y4 = ops.reshape(y3, (-1,))
            return y4

    class Net(Cell):
        def construct(self, x0, x1, x2, x3):
            y0 = NetFloatStatus(x0.shape)(x0)
            y1 = NetFloatStatus(x1.shape)(x1)
            y2 = NetFloatStatus(x2.shape)(x2)
            y3 = NetFloatStatus(x3.shape)(x3)
            y4 = ops.addn([y0, y1, y2, y3])
            return y4

    def get_output(enable_graph_kernel):
        jit_level = "O1" if enable_graph_kernel else "O0"
        context.set_context(jit_config={"jit_level": jit_level})
        x0 = Tensor(np.array([1.0, 2.0, np.inf]).astype(np.float32), input_type)
        x1 = Tensor(np.random.normal(0, 1, (3, 3, 1, 1)).astype(np.float32), mindspore.float32)
        x2 = Tensor(np.array([1.0, 2.1, np.nan]).astype(np.float32), input_type)
        x3 = Tensor(np.array([1.0, -np.inf, 0.0]).astype(np.float32), input_type)
        net = Net()
        y0 = net(x0, x1, x2, x3)
        return y0.asnumpy()

    context.set_context(mode=context.GRAPH_MODE)
    expect = get_output(False)
    expect = expect.astype(np.bool_)
    output = get_output(True)
    output = output.astype(np.bool_)
    assert np.allclose(expect, output, 0, 0)
