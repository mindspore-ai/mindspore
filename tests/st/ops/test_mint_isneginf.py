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
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP

import numpy as np
import pytest

import mindspore as ms
from mindspore import mint
from mindspore import context
from mindspore import Tensor

from mindspore.common import dtype as mstype


@test_utils.run_with_cell
def isneginf_forward(input_x):
    return mint.isneginf(input_x)


@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_2d_float32(context_mode):
    """
    Feature: mint ops Polar.
    Description: test polar forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})

    input_np = np.random.randn(2, 3).astype(np.float32)
    input_np[0][2] = float('inf')
    input_np[1][1] = -float('inf')

    input_x = Tensor(input_np, mstype.float32)
    output = isneginf_forward(input_x).asnumpy()
    expect = np.isneginf(input_np)
    assert np.array_equal(output, expect)

    input_np = np.random.randn(2, 3).astype(np.float16)
    input_np[0][2] = float('inf')
    input_np[1][1] = -float('inf')

    input_x = Tensor(input_np, mstype.float16)
    output = isneginf_forward(input_x).asnumpy()
    expect = np.isneginf(input_np)
    assert np.array_equal(output, expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_3d_int16(context_mode):
    """
    Feature: mint ops polar.
    Description: test polar forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    input_np = np.random.randn(3, 4).astype(np.int16)
    input_x = Tensor(input_np, mstype.int16)
    output = isneginf_forward(input_x).asnumpy()
    expect = np.isneginf(input_np)
    assert np.array_equal(output, expect)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
def test_forward_dynamic_shape():
    """
    Feature: mint.arange
    Description: Verify the result of arange forward with dynamic shape
    Expectation: success
    """
    inputs1_x = ms.Tensor(np.array([[1.0, 10, 2], [0, 6, 1]], np.float32))

    inputs2_x = ms.Tensor(np.array([[[5, 0.1], [0, 5.5]], [[0.1, 0.8], [5, 6]]], np.float32))

    TEST_OP(isneginf_forward, [[inputs1_x], [inputs2_x]],
            disable_mode=['GRAPH_MODE_GE'],
            case_config={'disable_grad': True})
