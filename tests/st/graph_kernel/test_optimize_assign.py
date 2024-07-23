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
from mindspore.ops import functional as F
from mindspore.common.parameter import Parameter


class TestOptAssignNet_1(Cell):
    def __init__(self):
        super(TestOptAssignNet_1, self).__init__()
        self.add = P.Add()
        self.reduce_max = P.ReduceMax()
        self.param = Parameter(
            Tensor(np.zeros([2, 2, 2]).astype(np.float32)), name='param')

    def construct(self, x, y):
        add_res = self.add(x, y)
        F.depend(add_res, F.assign(self.param, add_res))

        return self.reduce_max(add_res)


class TestOptAssignNet_2(Cell):
    def __init__(self):
        super(TestOptAssignNet_2, self).__init__()
        self.add = P.Add()
        self.param = Parameter(
            Tensor(np.zeros([2, 2, 2]).astype(np.float32)), name='param')

    def construct(self, x, y):
        add_res = self.add(x, y)
        F.depend(add_res, F.assign(self.param, add_res))

        return add_res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_opt_assign_output_1():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    net = TestOptAssignNet_1()
    result_open_gk = net(Tensor(input_x), Tensor(input_y))

    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=False)
    net_beta = TestOptAssignNet_1()
    result_close_gk = net_beta(Tensor(input_x), Tensor(input_y))
    res = np.allclose(result_open_gk.asnumpy(), result_close_gk.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_opt_assign_output_2():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    net = TestOptAssignNet_2()
    result_open_gk = net(Tensor(input_x), Tensor(input_y))

    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=False)
    net_beta = TestOptAssignNet_2()
    result_close_gk = net_beta(Tensor(input_x), Tensor(input_y))
    res = np.allclose(result_open_gk.asnumpy(), result_close_gk.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res
