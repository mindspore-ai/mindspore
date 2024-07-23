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

from tests.mark_utils import arg_mark
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops


class Net(Cell):
    """Net definition"""

    def __init__(self, x):
        super(Net, self).__init__()
        self.select = ops.Select()
        self.x = x

    def construct(self, cond, y):
        return self.select(cond, self.x, y)


def run(x):
    cond = Tensor([False])
    y = Tensor([3.0], mindspore.float32)
    expect_np = np.array([3.0]).astype(np.float32)
    net = Net(x)
    out = net(cond, y)
    assert np.allclose(expect_np, out.asnumpy(), 1.e-4, 1.e-7)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_finite():
    """
    Feature: test nan/inf case in graph kernel
    Description: input tensor is nan/inf with shape (1,)
    Expectation: no compile error
    """
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    x0 = Tensor(np.array([float('nan')]).astype(np.float32))
    run(x0)
    x1 = Tensor(np.array([float('inf')]).astype(np.float32))
    run(x1)
    x2 = Tensor(np.array([float('-inf')]).astype(np.float32))
    run(x2)
