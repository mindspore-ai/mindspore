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
import mindspore as ms
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def range_forward_func(start, limit, delta):
    return ops.range(start, limit, delta, maxlen=10)


@test_utils.run_with_cell
def range_backward_func(start, limit, delta):
    return ops.grad(range_forward_func, (0, 1, 2))(start, limit, delta)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_range_forward_tensor_input(mode):
    """
    Feature: range ops.
    Description: test ops range for Tensor input.
    Expectation: output a sequence of numbers that begins at "start" and extlimits by increments of "delta" up to but
    not including "limit".
    """
    context.set_context(mode=mode)
    start = ms.Tensor([0])
    limit = ms.Tensor([10])
    delta = ms.Tensor([2])
    output = range_forward_func(start, limit, delta)
    expect_output = np.array([0, 2, 4, 6, 8]).astype(np.int64)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_range_forward(mode):
    """
    Feature: range ops.
    Description: test ops range.
    Expectation: output a sequence of numbers that begins at "start" and extlimits by increments of "delta" up to but
    not including "limit".
    """
    context.set_context(mode=mode)
    start = 0
    limit = 10
    delta = 2
    output = range_forward_func(start, limit, delta)
    expect_output = np.array([0, 2, 4, 6, 8]).astype(np.int64)
    np.testing.assert_array_equal(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_range_backward(mode):
    """
    Feature: range ops.
    Description: test auto grad of ops range.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    start = 0
    limit = 10
    delta = 2
    output = range_backward_func(start, limit, delta)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_range_dynamic(mode):
    """
    Feature: range ops.
    Description: test ops range dynamic tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=mode)
    dyn_start = Tensor(shape=[None], dtype=mstype.int64)
    dyn_limit = Tensor(shape=[None], dtype=mstype.int64)
    dyn_delta = Tensor(shape=[None], dtype=mstype.int64)
    test_cell = test_utils.to_cell_obj(range_forward_func)
    test_cell.set_inputs(dyn_start, dyn_limit, dyn_delta)
    output1 = test_cell(Tensor([0], mstype.int64), Tensor([10], mstype.int64), Tensor([2], mstype.int64))
    expect_output1 = np.array([0, 2, 4, 6, 8]).astype(np.int64)
    np.testing.assert_array_equal(output1.asnumpy(), expect_output1)
    output2 = test_cell(Tensor([0], mstype.int64), Tensor([8], mstype.int64), Tensor([3], mstype.int64))
    expect_output2 = np.array([0, 3, 6]).astype(np.int64)
    np.testing.assert_array_equal(output2.asnumpy(), expect_output2)
