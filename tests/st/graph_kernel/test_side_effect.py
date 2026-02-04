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
import mindspore
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from tests.st.graph_kernel.gk_utils import AssertGKEnable
from tests.mark_utils import arg_mark


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.p1 = Parameter(Tensor([1.0], dtype=mindspore.float32), "p1")
        self.p2 = Parameter(Tensor([2.0], dtype=mindspore.float32), "p2")

    def construct(self, x):
        y0 = ops.assign(self.p1, x)
        ops.assign(self.p2, y0)
        y1 = ops.addn((x, self.p1, self.p2))
        return y1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
def test_assign_update():
    """
    Feature: test assign update
    Description: O1 assign update
    Expectation: the result match with the expected result
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O1"}, graph_kernel_flags="--enable_cluster_ops=Assign")
    with AssertGKEnable(True):
        net = Net()
        x0 = Tensor([3.0], dtype=mindspore.float32)
        out = net(x0)
        out = out.asnumpy()
    expect = np.array([9.0], dtype=np.float32)
    assert np.allclose(expect, out, 1e-4, 1e-4)
