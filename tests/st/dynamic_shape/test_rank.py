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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def rank_forward_func(x):
    return ops.operations.manually_defined.rank(x)


@test_utils.run_with_cell
def rank_backward_func(x):
    return ops.grad(rank_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_rank_forward(mode):
    """
    Feature: rank ops.
    Description: test ops rank.
    Expectation: output the right rank of a tensor.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    output = rank_forward_func(x)
    expect_output = 2
    np.testing.assert_equal(output, expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_rank_backward(mode):
    """
    Feature: rank ops.
    Description: test auto grad of ops rank.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    output = rank_backward_func(x)
    expect_output = np.array([[0, 0], [0, 0]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_rank_dynamic(mode):
    """
    Feature: rank ops.
    Description: test dynamic tensor rank.
    Expectation: output the right rank of a tensor.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.operations.manually_defined.rank)
    test_cell.set_inputs(x_dyn)
    x1 = Tensor(np.array([[2, 2],
                          [2, 2]]).astype(np.float32))
    output1 = test_cell(x1)
    expect_output1 = 2
    np.testing.assert_equal(output1, expect_output1)
    x2 = Tensor(np.array([[[1, 2],
                           [3, 4]],
                          [[5, 6],
                           [7, 8]]]).astype(np.float32))
    output2 = test_cell(x2)
    expect_output2 = 3
    np.testing.assert_equal(output2, expect_output2)
