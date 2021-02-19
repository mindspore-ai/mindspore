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
import pytest
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


def test_opt_assign_output_1():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    net = TestOptAssignNet_1()
    result_open_gk = net(Tensor(input_x), Tensor(input_y))

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=False, device_target="GPU")
    net_beta = TestOptAssignNet_1()
    result_close_gk = net_beta(Tensor(input_x), Tensor(input_y))
    res = np.allclose(result_open_gk.asnumpy(), result_close_gk.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


def test_opt_assign_output_2():
    np.random.seed(0)
    input_x = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)
    input_y = np.random.normal(0, 1, [2, 2, 2]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=True, device_target="GPU")
    net = TestOptAssignNet_2()
    result_open_gk = net(Tensor(input_x), Tensor(input_y))

    context.set_context(mode=context.GRAPH_MODE,
                        enable_graph_kernel=False, device_target="GPU")
    net_beta = TestOptAssignNet_2()
    result_close_gk = net_beta(Tensor(input_x), Tensor(input_y))
    res = np.allclose(result_open_gk.asnumpy(), result_close_gk.asnumpy(), rtol=1.e-4, atol=1.e-7, equal_nan=True)
    assert res


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_opt_assign_gpu_1():
    test_opt_assign_output_1()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_opt_assign_gpu_2():
    test_opt_assign_output_2()
