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
"""Tests for UpsampleBicubic2D and its grad under dynamic shape cases."""

import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor
from mindspore import ops
from mindspore.device_context.cpu.op_tuning import threads_num


@test_utils.run_with_cell
def upsample_bicubic2d_forward_func(x, size=None, scale_factor=None, align_corners=False):
    return ops.function.nn_func.interpolate_ext(x, size, scale_factor, "bicubic", align_corners)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
def test_upsample_bicubic_2d_size_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test op UpsampleBicubic2D and UpsampleBicubic2DGrad.
    Expectation: expect UpsampleBicubic2D and UpsampleBicubic2DGrad result.
    """
    threads_num(1)  # multi-threads have none-initialized bug now.
    input_case1 = Tensor(np.random.randn(2, 5, 60, 30), dtype=ms.float32)
    input_case2 = Tensor(np.random.randn(4, 3, 15, 10), dtype=ms.float32)
    TEST_OP(
        upsample_bicubic2d_forward_func,
        [
            [input_case1, (100, 200), None, True],
            [input_case2, (40, 80), None, False],
        ],
        disable_mode=["GRAPH_MODE_GE"],
        disable_case=['EmptyTensor', 'ScalarTensor', 'Deterministic'],
        case_config={'disable_input_check': True}
    )
