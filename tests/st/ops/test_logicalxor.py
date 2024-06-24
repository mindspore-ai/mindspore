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

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore import ops
import tests.st.utils.test_utils as test_utils


@test_utils.run_with_cell
def forward_func(x, y):
    return ops.logical_xor(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_logicalxor_float32(context_mode):
    """
    Feature: logicalxor
    Description: test logicalxor
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    x = Tensor([True, True, False], ms.bool_)
    y = Tensor([False, False, False], ms.bool_)
    output = forward_func(x, y)
    expected = np.array([True, True, False], np.bool_)
    assert np.all(output.asnumpy() == expected)
