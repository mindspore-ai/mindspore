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
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import context, ops
from mindspore.device_context.cpu.op_tuning import threads_num


def set_mode(mode):
    if mode == "GRAPH_MODE_O0":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    elif mode == "GRAPH_MODE":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@test_utils.run_with_cell
def upsample_linear1d_forward_func(x, size=None, scale_factor=None, align_corners=False):
    return ops.function.nn_func.interpolate_ext(x, size, scale_factor, "linear", align_corners)


@test_utils.run_with_cell
def upsample_linear1d_backward_func(x, size=None, scale_factor=None, align_corners=False):
    return ms.grad(upsample_linear1d_forward_func, (0,))(x, size, scale_factor, align_corners)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["GRAPH_MODE_O0", "PYNATIVE_MODE"])
def test_upsample_linear_1d(mode):
    """
    Feature: test ops.
    Description: test op UpsampleLinear1D.
    Expectation: success.
    """
    set_mode(mode)

    input_tensor = Tensor(
        np.array([[[0.1, 0.3, 0.5], [0.7, 0.9, 1.1]]]), dtype=ms.float32
    )
    expected = np.array(
        [[[0.1, 0.18, 0.26, 0.34, 0.42, 0.5], [0.7, 0.78, 0.86, 0.94, 1.02, 1.1]]]
    ).astype(np.float32)
    error = np.ones(shape=expected.shape) * 1.0e-4
    out = upsample_linear1d_forward_func(input_tensor, (6,), None, True)
    diff = abs(out.asnumpy() - expected)
    assert np.all(diff < error)

    out = upsample_linear1d_backward_func(input_tensor, (6), None, False)
    expected = np.array([[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]).astype(np.float32)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    out = upsample_linear1d_backward_func(input_tensor, None, (2.3,), True)
    expected = np.array([[[1.8, 2.4, 1.8], [1.8, 2.4, 1.8]]]).astype(np.float32)
    diff = abs(out.asnumpy() - expected)
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_upsample_linear_1d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleLinear1D and UpsampleLinear1DGrad.
    Expectation: expect UpsampleLinear1D and UpsampleLinear1DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 10), dtype=ms.float32)
    TEST_OP(
        upsample_linear1d_forward_func,
        [
            [input_case1, (100,), None, True],
            [input_case2, (40,), None, False],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
def test_upsample_linear_1d_scales_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleLinear1D and UpsampleLinear1DGrad.
    Expectation: expect UpsampleLinear1D and UpsampleLinear1DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 10), dtype=ms.float32)
    TEST_OP(
        upsample_linear1d_forward_func,
        [
            [input_case1, None, (2.6,), True],
            [input_case2, None, (3.7,), True],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )
