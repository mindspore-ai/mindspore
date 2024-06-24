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

import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, context, mint


@test_utils.run_with_cell
def upsample_linear1d_forward_func(x, size=None, scale_factor=None, align_corners=False):
    return mint.nn.functional.interpolate(x, size, scale_factor, "linear", align_corners)


@test_utils.run_with_cell
def upsample_linear1d_backward_func(x, size=None, scale_factor=None, align_corners=False):
    return ops.grad(upsample_linear1d_forward_func, (0,))(x, size, scale_factor, align_corners)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_upsample_linear_1d(mode):
    """
    Feature: test ops.
    Description: test op UpsampleLinear1D.
    Expectation: success.
    """
    context.set_context(mode=mode)

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

    out = upsample_linear1d_forward_func(input_tensor, None, (2.3,), False)
    expected = np.array(
        [[[0.1, 0.1304, 0.2174, 0.3043, 0.3913, 0.4783],
          [0.7, 0.7304, 0.8174, 0.9043, 0.9913, 1.0783]]]).astype(np.float32)
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_upsample_linear_1d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleLinear1D and UpsampleLinear1DGrad.
    Expectation: expect UpsampleLinear1D and UpsampleLinear1DGrad result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 10), dtype=ms.float32)
    TEST_OP(
        upsample_linear1d_forward_func,
        [
            [input_case1, (100,), None, True],
            [input_case2, (40,), None, False],
        ],
        'upsample_linear1d',
        disable_input_check=True
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_upsample_linear_1d_scales_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleLinear1D and UpsampleLinear1DGrad.
    Expectation: expect UpsampleLinear1D and UpsampleLinear1DGrad result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 10), dtype=ms.float32)
    TEST_OP(
        upsample_linear1d_forward_func,
        [
            [input_case1, None, (2.6,), True],
            [input_case2, None, (3.7,), True],
        ],
        'upsample_linear1d',
        disable_input_check=True
    )
