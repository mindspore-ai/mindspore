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

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def resize_nearest_neighbor_forward_func(x, size, align_corners):
    return ops.ResizeNearestNeighbor(size, align_corners)(x)


@test_utils.run_with_cell
def resize_nearest_neighbor_backward_func(x, size, align_corners):
    return ops.grad(resize_nearest_neighbor_forward_func, (0, 1))(x, size, align_corners)


def resize_nearest_neighbor_dyn_shape_func(x, size, align_corners):
    return ops.ResizeNearestNeighbor(size, align_corners)(x)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_resize_nearest_neighbor_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op resize_nearest_neighbor forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]).astype(data_type))
    size = (2, 2)
    align_corners = True
    out = resize_nearest_neighbor_forward_func(x, size, align_corners)
    expect_out = np.array([[[[-0.1, 3.6], [0.4, -3.2]]]]).astype(data_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_resize_nearest_neighbor_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op resize_nearest_neighbor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([[[[1., 2., 3.], [1., 2., 3.]]]]).astype(data_type))
    size = (2, 2)
    align_corners = True
    grads = resize_nearest_neighbor_backward_func(x, size, align_corners)
    expect_out = np.array([[[[1., 0., 1.], [1., 0., 1.]]]]).astype(np.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
def test_resize_nearest_neighbor_op_dynamic(context_mode):
    """
    Feature: test dynamic tensor of resize_nearest_neighbor
    Description: test ops resize_nearest_neighbor dynamic tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x1 = ms.Tensor(np.array([[[[1., 2., 3.], [1., 2., 3.]]]]).astype(np.float32))
    size = (2, 2)
    align_corners = True
    test_cell = test_utils.to_cell_obj(resize_nearest_neighbor_dyn_shape_func)
    test_cell.set_inputs(x_dyn, size, align_corners)
    output = test_cell(x1, size, align_corners)
    expect_out = resize_nearest_neighbor_forward_func(x1, size, align_corners)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-4)

    x2 = ms.Tensor(np.array([[[[2., 2., 3.], [1., 3., 3.]]]]).astype(np.float32))
    output = test_cell(x2, size, align_corners)
    expect_out = resize_nearest_neighbor_forward_func(x2, size, align_corners)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-4)
