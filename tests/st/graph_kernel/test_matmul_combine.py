# Copyright 2023 Huawei Technologies Co., Ltd
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


class CustomNet(Cell):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.matmul = P.MatMul()
        self.add = P.Add()
        self.abs = P.Abs()

    def construct(self, mx_a, mx_b1, bias1, mx_b2, bias2):
        # use Abs to create a shared input for matmul
        abs1 = P.Abs()(mx_a)
        abs2 = P.Abs()(mx_a)

        # branch 1: matmul - add - abs
        m1 = self.matmul(abs1, mx_b1)
        m1 = self.add(m1, bias1)
        m1 = self.abs(m1)

        # branch 2: matmul - add - abs
        m2 = self.matmul(abs2, mx_b2)
        m2 = self.add(m2, bias2)
        m2 = self.abs(m2)
        return m1, m2


def get_output(i0, i1, i2, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel, save_graphs=False)
    net = CustomNet()
    mx_a = i0
    mx_b1 = i1 * 3
    mx_b2 = i1 * 2
    bias1 = i2 * 3
    bias2 = i2 * 2
    output = net(mx_a, mx_b1, bias1, mx_b2, bias2)
    return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parallel_matmul_combine():
    """
    Feature: Parallel Matmul combination
    Description: on GPU device
    Expectation: network return same result with the feature on and off
    """
    context.set_context(mode=context.GRAPH_MODE)
    i0 = Tensor(np.random.normal(1, 0.01, [96, 800]).astype(np.float16))
    i1 = Tensor(np.random.normal(1, 0.01, [800, 128]).astype(np.float16))
    i2 = Tensor(np.random.normal(1, 0.01, [1, 128]).astype(np.float16))

    expect = get_output(i0, i1, i2, False)
    output = get_output(i0, i1, i2, True)
    for exp, out in zip(expect, output):
        expect_np = exp.asnumpy().copy()
        output_np = out.asnumpy().copy()
        assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)
