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
    def __init__(self, shape):
        super(Net, self).__init__()
        self.param = Parameter(Tensor(np.ones(shape), dtype=mindspore.bfloat16), "param")

    def construct(self, x0, x1):
        y0 = ops.Abs()(x0)
        y1 = ops.Add()(x0, y0)
        y2 = ops.Cast()(x1, mindspore.bfloat16)
        y3 = ops.Abs()(y2)
        y4 = ops.AddN()((y1, y2, y3))
        y5 = ops.Assign()(self.param, y4)
        return y0, y5


def get_output(x0, x1, shape, enable_graph_kernel):
    jit_level = "O1" if enable_graph_kernel else "O0"
    context.set_context(jit_config={"jit_level": jit_level})
    with AssertGKEnable(enable_graph_kernel):
        net = Net(shape)
        y0, _ = net(x0, x1)
    return y0.float().asnumpy(), net.param.float().asnumpy()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_convert_bfloat16():
    """
    Feature: test graph kernel bfloat16 data type
    Description: input is bfloat16
    Expectation: the result match with the expected result
    """
    context.set_context(mode=context.GRAPH_MODE)
    np.random.seed(1)
    shape = (4, 4)
    x0 = np.random.normal(0, 1, shape).astype(np.float32)
    x1 = np.abs(np.random.normal(0, 1, shape).astype(np.float32))
    x0_ms = Tensor(x0, mindspore.bfloat16)
    x1_ms = Tensor(x1)
    expects = get_output(x0_ms, x1_ms, shape, False)
    outputs = get_output(x0_ms, x1_ms, shape, True)
    compare_result = [np.allclose(e, o, 1.5e-3, 1.5e-3) for e, o in zip(expects, outputs)]
    assert False not in compare_result
