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
def upsample_bilinear2d_forward_func(x, size=None, scale_factor=None, align_corners=False):
    return mint.nn.functional.interpolate(x, size, scale_factor, "bilinear", align_corners)


@test_utils.run_with_cell
def upsample_bilinear2d_backward_func(x, size=None, scale_factor=None, align_corners=False):
    return ops.grad(upsample_bilinear2d_forward_func, (0,))(x, size, scale_factor, align_corners)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_upsample_bilinear_2d(mode):
    """
    Feature: test ops.
    Description: test op UpsampleBillinear2D.
    Expectation: success.
    """
    context.set_context(mode=mode)
    input_tensor = Tensor(
        np.array(
            [[[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]]]
        ).astype(np.float32)
    )
    expected = np.array(
        [
            [
                [
                    [0.1000, 0.1500, 0.2000, 0.2500, 0.3000],
                    [0.2000, 0.2500, 0.3000, 0.3500, 0.4000],
                    [0.3000, 0.3500, 0.4000, 0.4500, 0.5000],
                    [0.4000, 0.4500, 0.5000, 0.5500, 0.6000],
                ],
                [
                    [0.7000, 0.7500, 0.8000, 0.8500, 0.9000],
                    [0.8000, 0.8500, 0.9000, 0.9500, 1.0000],
                    [0.9000, 0.9500, 1.0000, 1.0500, 1.1000],
                    [1.0000, 1.0500, 1.1000, 1.1500, 1.2000],
                ],
            ]
        ]
    ).astype(np.float32)
    out = upsample_bilinear2d_forward_func(input_tensor, (4, 5), None, True)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)

    expected = np.array(
        [
            [
                [[3.0000, 4.0000, 3.0000], [3.0000, 4.0000, 3.0000]],
                [[3.0000, 4.0000, 3.0000], [3.0000, 4.0000, 3.0000]],
            ]
        ]
    ).astype(np.float32)
    out = upsample_bilinear2d_backward_func(input_tensor, (4, 5), None, True)
    diff = abs(out.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_upsample_bilinear_2d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleBilinear2D and UpsampleBilinear2DGrad.
    Expectation: expect UpsampleBilinear2D and UpsampleBilinear2DGrad result.
    """
    ms.context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10), dtype=ms.float32)
    TEST_OP(
        upsample_bilinear2d_forward_func,
        [
            [input_case1, (100, 200), None, True],
            [input_case2, (40, 80), None, False],
        ],
        'upsample_bilinear2d',
        disable_input_check=True
    )
