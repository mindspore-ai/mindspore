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
        self.sqrt = P.Sqrt()
        self.add = P.Add()
        self.neg = P.Neg()
        self.mul = P.Mul()

    def construct(self, x0, x1):
        sqrt_res = self.sqrt(x0)
        neg_res = self.neg(sqrt_res)
        add_res = self.add(x1, sqrt_res)
        real_res = self.mul(add_res, add_res)
        return neg_res, real_res


def easy_fuse():
    def get_output(i0, i1, enable_graph_kernel=False):
        context.set_context(enable_graph_kernel=enable_graph_kernel)
        net_obj = Net()
        output = net_obj(i0, i1)
        return output
    i0 = Tensor(np.random.uniform(1, 2, [1, 1024]).astype(np.float32))
    i1 = Tensor(np.random.uniform(1, 2, [1024, 1024]).astype(np.float32))
    expect = get_output(i0, i1, False)
    output = get_output(i0, i1, True)
    expect0_np = expect[0].asnumpy().copy()
    expect1_np = expect[1].asnumpy().copy()
    output0_np = output[0].asnumpy().copy()
    output1_np = output[1].asnumpy().copy()
    assert np.allclose(expect0_np, output0_np, rtol=1.e-4, atol=1.e-4, equal_nan=True)
    assert np.allclose(expect1_np, output1_np, rtol=1.e-4, atol=1.e-4, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_easy_fuse_cpu():
    """
    Feature: easy test case for graph_kernel in cpu.
    Description: cpu test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    context.set_context(mode=context.GRAPH_MODE)
    easy_fuse()
