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

import pytest
from tests.mark_utils import arg_mark
import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import DataType, CustomRegOp
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class CustomNet(Cell):
    def __init__(self):
        super(CustomNet, self).__init__()
        aclop_ref_info = CustomRegOp("AddCustom") \
            .input(0, "x", "required") \
            .input(1, "y", "required") \
            .output(0, "z", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.custom_add = ops.Custom("AddCustom", lambda x, _: x, lambda x, _: x, func_type="aot",
                                     reg_info=aclop_ref_info)
        self.custom_add.add_prim_attr("custom_aclop", True)
        self.add = P.Add()
        self.sub = P.Sub()

    def construct(self, x, y, z):
        res = self.add(x, y)
        res = self.custom_add(res, y)
        res = self.sub(res, z)
        return res


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_add_aclop(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for AddCustom
    Expectation: the result match with numpy result
    """
    context.set_context(jit_level='O0')

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs")
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048)
    net = CustomNet()
    expect_out = x + y + y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
def test_custom_add_aclop_dynamic(context_mode):
    """
    Feature: Custom op testcase
    Description: test case for AddCustom op in dynamic shape
    Expectation: the result match with numpy result
    """
    context.set_context(jit_level='O0')

    context.set_context(mode=context_mode, save_graphs=False, save_graphs_path="./graphs")
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048)
    dyn_x = Tensor(shape=(8, None), dtype=mstype.float16)
    net = CustomNet()
    expect_out = x + y + y - z
    net.set_inputs(dyn_x, Tensor(y), Tensor(z))
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
def test_custom_add_aclop_graph():
    """
    Feature: Custom op testcase
    Description: test case for AddCustom op in graph mode
    Expectation: the result match with numpy result
    """
    context.set_context(jit_level='O2')
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, save_graphs_path="./graphs")
    x = np.ones([8, 2048]).astype(np.float16)
    y = np.ones([8, 2048]).astype(np.float16)
    z = np.random.rand(8, 2048).astype(np.float16)
    net = CustomNet()
    expect_out = x + y + y - z
    out = net(Tensor(x), Tensor(y), Tensor(z))
    assert np.allclose(out.asnumpy(), expect_out, 0.001, 0.001)
