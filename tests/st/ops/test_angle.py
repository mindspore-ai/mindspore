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
def angle_forward_func(x):
    return ops.angle(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_angle(context_mode):
    """
    Feature: angle
    Description: test angle
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    x = Tensor([-1.5 + 7.8j, 3 + 5.75j], ms.complex64)
    output = angle_forward_func(x)
    expected = np.array([1.7607845, 1.0899091], np.float32)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-3)
