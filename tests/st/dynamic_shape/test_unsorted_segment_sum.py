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
# pylint: disable=unused-variable
import pytest
import numpy as np
from tests.st.utils import test_utils
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def unsorted_segment_forward_func(x, y, z):
    return ops.unsorted_segment_sum(x, y, z)


@test_utils.run_with_cell
def unsorted_segment_backward_func(x, y, z):
    return ops.grad(unsorted_segment_forward_func, (0, 1, 2))(x, y, z)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_unsorted_segment_forward(mode):
    """
    Feature: assign ops.
    Description: test ops assign.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 0, 1, 2]).astype(np.int32))
    num_segments = 4
    output = unsorted_segment_forward_func(x, segment_ids, num_segments)
    expected = np.asarray([3., 3., 4., 0.]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_unsorted_segment_backward(mode):
    """
    Feature: assign ops.
    Description: test auto grad of ops assign.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1, 2, 3, 4]).astype(np.float32))
    segment_ids = Tensor(np.array([0, 0, 1, 2]).astype(np.int32))
    num_segments = 4
    dx = unsorted_segment_backward_func(x, segment_ids, num_segments)
    except_dvariable = np.asarray([1., 1., 1., 1.]).astype(np.float32)
    np.testing.assert_array_almost_equal(dx[0].asnumpy(), except_dvariable, decimal=4)
