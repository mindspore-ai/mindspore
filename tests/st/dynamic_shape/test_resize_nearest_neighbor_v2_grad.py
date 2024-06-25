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
def resize_nearest_neighbor_v2_grad_forward_func(x, size, align_corners, half_pixel_centers):
    return ops.auto_generate.ResizeNearestNeighborV2Grad(align_corners, half_pixel_centers)(x, size)


def resize_nearest_neighbor_v2_grad_dyn_shape_func(x, size, align_corners, half_pixel_centers):
    return ops.auto_generate.ResizeNearestNeighborV2Grad(align_corners, half_pixel_centers)(x, size)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_resize_nearest_neighbor_v2_grad_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op resize_nearest_neighbor_v2_grad forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]).astype(data_type))
    size = (2, 2)
    align_corners = False
    half_pixel_centers = False
    out = resize_nearest_neighbor_v2_grad_forward_func(x, size, align_corners, half_pixel_centers)
    expect_out = np.array([[[[0.2, 3.6], [0.9, -3.2]]]]).astype(data_type)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
def test_resize_nearest_neighbor_v2_grad_op_dynamic(context_mode):
    """
    Feature: test dynamic tensor of resize_nearest_neighbor_v2_grad
    Description: test ops resize_nearest_neighbor_grad dynamic tensor input.
    Expectation: output the right result.
    """
    ms.context.set_context(mode=context_mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x1 = ms.Tensor(np.array([[[[1., 2., 3.], [1., 2., 3.]]]]).astype(np.float32))
    size = (2, 2)
    align_corners = False
    half_pixel_centers = False
    test_cell = test_utils.to_cell_obj(resize_nearest_neighbor_v2_grad_dyn_shape_func)
    test_cell.set_inputs(x_dyn, size, align_corners, half_pixel_centers)
    output = test_cell(x1, size, align_corners, half_pixel_centers)
    expect_out = resize_nearest_neighbor_v2_grad_forward_func(x1, size, align_corners, half_pixel_centers)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-4)

    x2 = ms.Tensor(np.array([[[[2., 2., 3.], [3., 2., 3.]]]]).astype(np.float32))
    output = test_cell(x2, size, align_corners, half_pixel_centers)
    expect_out = resize_nearest_neighbor_v2_grad_forward_func(x2, size, align_corners, half_pixel_centers)
    np.testing.assert_allclose(output.asnumpy(), expect_out.asnumpy(), rtol=1e-4)
