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
from tests.st.utils import test_utils
import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def reciprocal_grad_func(y, dy):
    return ops.auto_generate.ReciprocalGrad()(y, dy)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reciprocal_grad(mode):
    """
    Feature: reciprocal_grad ops.
    Description: test ops reciprocal.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    y = Tensor(np.array([1., 0.5, 0.25]).astype(np.float32))
    dy = Tensor(np.array([1., 1., 1.]).astype(np.float32))
    output = reciprocal_grad_func(y, dy)
    expect_output = np.asarray([-1., -0.25, -0.0625]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reciprocal_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reciprocal_grad ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    y = Tensor(np.array([[[1., 0.5, 0.25]]]).astype(np.float32))
    dy = Tensor(np.array([[[1., 1., 1.]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(reciprocal_grad_func))
    output = nest_vmap(y, dy)
    expect_out = reciprocal_grad_func(y, dy)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reciprocal_grad_dynamic(mode):
    """
    Feature: reciprocal_grad ops.
    Description: test ops reciprocal_grad dynamic tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.ReciprocalGrad())
    test_cell.set_inputs(y_dyn, dy_dyn)
    y1 = Tensor(np.array([1., 0.5, 0.25]).astype(np.float32))
    dy1 = Tensor(np.array([1., 1., 1.]).astype(np.float32))
    output1 = test_cell(y1, dy1)
    expect_output1 = np.asarray([-1., -0.25, -0.0625]).astype(np.float32)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect_output1, decimal=4)
    y2 = Tensor(np.array([[1.0, 0.5],
                          [0.25, 0.2]]).astype(np.float32))
    dy2 = Tensor(np.array([[1., 1.],
                           [1., 1.]]).astype(np.float32))
    output2 = test_cell(y2, dy2)
    expect_output2 = np.asarray([[-1., -0.25],
                                 [-0.0625, -0.04]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect_output2, decimal=4)
