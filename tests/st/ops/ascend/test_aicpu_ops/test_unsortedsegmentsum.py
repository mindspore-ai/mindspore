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

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore import ops
import tests.st.utils.test_utils as test_utils


@test_utils.run_with_cell
def forward_func(input_x, segment_ids, num_segments):
    return ops.unsorted_segment_sum(input_x, segment_ids, num_segments)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu_training
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_unsortedsegmentsum(context_mode):
    """
    Feature: unsortedsegmentsum
    Description: test unsortedsegmentsum
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    input_x = Tensor([1, 2, 3, 4], ms.float64)
    segment_ids = Tensor([0, 0, 1, 2], ms.int32)
    num_segments = 4
    output = forward_func(input_x, segment_ids, num_segments)
    expected = np.array([3., 3., 4., 0.], dtype=np.float64)
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-3)
