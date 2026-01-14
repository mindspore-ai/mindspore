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
import pytest
import numpy as np
import mindspore as ms
from mindspore.mint import nansum
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import tests.st.utils.test_utils as test_utils


@test_utils.run_with_cell
def nansum_forward_func(x, dim, keepdim, dtype=None):
    return nansum(x, dim, keepdim=keepdim, dtype=dtype)


@test_utils.run_with_cell
def nansum_backward_func(x, dim, keepdim, dtype=None):
    return ms.grad(nansum_forward_func, (0))(x,
                                             dim,
                                             keepdim=keepdim,
                                             dtype=dtype)


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_nansum(mode):
    """
    Feature: ops.nansum
    Description: Verify the result of nansum
    Expectation: success
    """
    set_mode(mode)
    x = ms.Tensor([[float("nan"), 128.1, -256.9],
                   [float("nan"), float("nan"), 128]], ms.float32)
    output = nansum_forward_func(x, 0, keepdim=True)
    expect_output = [[0, 128.1, -128.9]]
    assert np.allclose(output.asnumpy(), expect_output)

    backward_output = nansum_backward_func(x, 0, keepdim=True)
    backward_expect_output = [[0., 1., 1.], [0., 0., 1.]]
    assert np.allclose(backward_output.asnumpy(), backward_expect_output)


@arg_mark(
    plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=4.01
)
def test_mint_nansum_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function nansum forward with dynamic shape.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor([[float("nan"), 128.1, -256.9],
                        [float("nan"), float("nan"), 128]], ms.float32)
    axis1 = 1
    keepdim1 = False
    input2 = ms.Tensor([[[float("nan"), 128.1, -256.9],
                         [float("nan"), float("nan"), 128],
                         [float("nan"), float("nan"), 128]]], ms.float32)
    axis2 = 0
    keepdim2 = True
    TEST_OP(nansum_forward_func,
            [[input1, axis1, keepdim1], [input2, axis2, keepdim2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['ScalarTensor'])
