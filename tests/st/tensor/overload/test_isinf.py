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

"""Test the overload tensor_method.tensor_isinf."""

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils

import numpy as np
import pytest

from mindspore import context
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import tensor_method as tm


@test_utils.run_with_cell
def isinf_forward(x):
    return x.isinf()


@test_utils.run_with_cell
def tensor_method_isinf_forward(x):
    return tm.tensor_isinf(x)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential'
)
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_isinf(context_mode):
    """
    Feature: Tensor.isinf.
    Description: test isinf forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, jit_config={"jit_level": "O0"})
    input_np = np.random.randn(3, 3).astype(np.float16)
    input_np[2][1] = float('inf')
    input_np[1][1] = -float('inf')

    input_x = Tensor(input_np, mstype.float16)
    output = isinf_forward(input_x).asnumpy()
    output_tensor_method = tensor_method_isinf_forward(input_x).asnumpy()
    expect = np.isinf(input_np)
    assert np.array_equal(output, expect)
    assert np.array_equal(output_tensor_method, expect)
