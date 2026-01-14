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
        self.matmul = P.MatMul(transpose_a=True, transpose_b=True)

    def construct(self, x, y):
        return self.matmul(x, y)


def get_output(i0, i1, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(jit_level='O1')
        context.set_context(
            graph_kernel_flags="--enable_cluster_ops=MatMul --online_tuning=1")
    else:
        context.set_context(jit_level='O0')
    net = Net()
    output = net(i0, i1)
    return output


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=30.0)
def test_online_tuning():
    """
    Feature: test online tuning matmul case for graph_kernel in Ascend.
    Description: ascend test case, use graph_kernel execute ops.
    Expectation: the result match with close graph_kernel result
    """
    context.set_context(mode=context.GRAPH_MODE)
    i0 = Tensor(np.random.normal(1, 0.01, [8192, 512]).astype(np.float16))
    i1 = Tensor(np.random.normal(1, 0.01, [1024, 8192]).astype(np.float16))
    expect = get_output(i0, i1, False)
    output = get_output(i0, i1, True)
    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()
    assert np.allclose(expect_np, output_np, 1e-3, 1e-3)
