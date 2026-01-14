# Copyright 2025 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.ops as ops
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from tests.st.graph_kernel.gk_utils import AssertGKEnable
from tests.mark_utils import arg_mark


class Net(Cell):
    def construct(self, x0, x1, rounding_mode):
        y0 = ops.auto_generate.DivMod()(x0, x1, rounding_mode)
        return y0


def get_output(x0, x1, rounding_mode, enable_graph_kernel):
    jit_level = "O1" if enable_graph_kernel else "O0"
    context.set_context(jit_config={"jit_level": jit_level})
    with AssertGKEnable(enable_graph_kernel):
        net = Net()
        output = net(x0, x1, rounding_mode)
    output = output.float().asnumpy() if output.dtype == mindspore.bfloat16 else output.asnumpy()
    return output


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize("data_type", [mindspore.float16, mindspore.float32, mindspore.bfloat16])
@pytest.mark.parametrize("rounding_mode", [None, "floor", "trunc"])
def test_div_mod(data_type, rounding_mode):
    """
    Feature: test graph kernel DivMod precision
    Description: test op with different data types
    Expectation: the result match with the expected result
    """
    context.set_context(mode=context.GRAPH_MODE)
    shape = (4, 8)
    x0 = np.random.normal(0, 1, shape).astype(np.float32)
    x1 = np.random.normal(0, 1, shape).astype(np.float32)
    x0_ms = Tensor(x0, data_type)
    x1_ms = Tensor(x1, data_type)
    expect = get_output(x0_ms, x1_ms, rounding_mode, False)
    output = get_output(x0_ms, x1_ms, rounding_mode, True)
    np.testing.assert_allclose(expect, output, 0, 0)
