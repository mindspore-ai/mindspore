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

from tests.mark_utils import arg_mark
import numpy as np
import mindspore
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.const0 = Tensor([2], dtype=mindspore.int64)
    def construct(self, para0, para1):
        y0 = ops.Add()(para0, para1)
        y1 = ops.Cast()(y0, mindspore.float32)
        y2 = ops.Mul()(y1, y1)
        y3 = ops.ReduceSum(keep_dims=True, skip_mode=False)(y2, self.const0)
        return y3, y1, y0


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dvm_dynamic_shape_codegen():
    """
    Feature: Test the functionality of O1 CodeGen
    Description: test case for O1 CodeGen under dynamic shape situation
    Expectation: the process runs normally without any error or exception, and the results of O0 and O1 are the same
    """
    np.random.seed(42)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    para0_dyn = Tensor(shape=(1, None, 4096), dtype=mindspore.float16)
    para1_dyn = Tensor(shape=(1, None, 4096), dtype=mindspore.float16)

    # Generate 5 groups of inputs. Do NOT modify the range.
    data0 = []
    data1 = []
    for i in range(2686, 2691):
        data0.append(np.random.normal(0, 1, (1, i, 4096)).astype(np.float16))
        data1.append(np.random.normal(0, 1, (1, i, 4096)).astype(np.float16))

    outputs_o0 = []
    outputs_o1 = []
    # compute the results of O0
    net_o0 = Net()
    net_o0.set_inputs(para0_dyn, para1_dyn)
    context.set_context(jit_level='O0')
    for i in range(5):
        arg0 = Tensor(data0[i], dtype=mindspore.float16)
        arg1 = Tensor(data1[i], dtype=mindspore.float16)
        out = net_o0(arg0, arg1)
        for o in out:
            outputs_o0.append(o.asnumpy())

    # compute the results of O1
    net_o1 = Net()
    net_o1.set_inputs(para0_dyn, para1_dyn)
    context.set_context(jit_level='O1')
    for i in range(5):
        arg0 = Tensor(data0[i], dtype=mindspore.float16)
        arg1 = Tensor(data1[i], dtype=mindspore.float16)
        out = net_o1(arg0, arg1)
        for o in out:
            outputs_o1.append(o.asnumpy())


    # compare results between O0 and O1
    for i in range(15):
        assert np.allclose(outputs_o0[i], outputs_o1[i], 2e-3)
