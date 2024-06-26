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

import numpy as np
import pytest
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from mindspore import ops
import mindspore as ms


@test_utils.run_with_cell
def indexput_forward_func(x1, x2, indices, accumulate=0):
    return ops.IndexPut(accumulate)(x1, x2, indices)


@test_utils.run_with_cell
def indexput_backward_func(x1, x2, indices, accumulate=0):
    return ops.grad(indexput_forward_func, (0, 1))(x1, x2, indices, accumulate)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_indexput_op_forward(context_mode):
    """
    Feature: Ops.
    Description: test op indexput forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x1 = ms.Tensor([[0, 0], [0, 0]], dtype=ms.float32)
    x2 = ms.Tensor([1, 2, 3, 4], dtype=ms.float32)
    indices = (ms.Tensor([0, 0, 1, 1]), ms.Tensor([0, 1, 0, 1]))
    out = indexput_forward_func(x1, x2, indices)
    expected_out = np.array([[1, 2], [3, 4]], np.float32)
    np.testing.assert_allclose(out.asnumpy(), expected_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_indexput_op_backward(context_mode):
    """
    Feature: Ops.
    Description: test op indexput backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x1 = ms.Tensor([[0, 0], [0, 0]], dtype=ms.float32)
    x2 = ms.Tensor([1, 2, 3, 4], dtype=ms.float32)
    indices = (ms.Tensor([0, 0, 1, 1]), ms.Tensor([0, 1, 0, 1]))
    out = indexput_backward_func(x1, x2, indices)
    x1_grad, x2_grad = out[0].asnumpy(), out[1].asnumpy()
    expected_x1_grad = np.zeros(x1.shape)
    expected_x2_grad = np.ones(x2.shape)
    np.testing.assert_allclose(x1_grad, expected_x1_grad, rtol=1e-3)
    np.testing.assert_allclose(x2_grad, expected_x2_grad, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_indexput_op_forward_accumulate(context_mode):
    """
    Feature: Ops.
    Description: test op indexput forward with accumulation.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)

    x1 = ms.Tensor([[1, 1], [1, 1]], dtype=ms.float32)
    x2 = ms.Tensor([1, 2, 3, 4], dtype=ms.float32)
    indices = (ms.Tensor([0, 0, 1, 1]), ms.Tensor([0, 1, 0, 1]))
    out = indexput_forward_func(x1, x2, indices, 1)
    expected_out = np.array([[2, 3], [4, 5]], np.float32)
    np.testing.assert_allclose(out.asnumpy(), expected_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_indexput_op_backward_accumulate(context_mode):
    """
    Feature: Ops.
    Description: test op indexput backward with accumulation.
    Expectation: expect correct result.
    """
    x1 = ms.Tensor([[1, 1], [1, 1]], dtype=ms.float32)
    x2 = ms.Tensor([1, 2, 3, 4], dtype=ms.float32)
    indices = (ms.Tensor([0, 0, 1, 1]), ms.Tensor([0, 1, 0, 1]))
    out = indexput_backward_func(x1, x2, indices, 1)
    x1_grad, x2_grad = out[0].asnumpy(), out[1].asnumpy()
    expected_x1_grad = np.ones(x1.shape)
    expected_x2_grad = np.ones(x2.shape)
    np.testing.assert_allclose(x1_grad, expected_x1_grad, rtol=1e-3)
    np.testing.assert_allclose(x2_grad, expected_x2_grad, rtol=1e-3)
