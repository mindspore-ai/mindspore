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
from mindspore import Tensor, context
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def prelu_grad_func(y, x, weight):
    return ops.auto_generate.PReLUGrad()(y, x, weight)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_prelu_grad(mode):
    """
    Feature: prelu_grad ops.
    Description: test ops prelu_grad.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    dy = Tensor(np.array([[[-0.6, -0.5],
                           [-2.4, -1.8],
                           [0.6, 0.3]],
                          [[0., 1.],
                           [2., 3.],
                           [4., 5.]]]).astype(np.float32))
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output = prelu_grad_func(dy, x, weight)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_grad_dynamic(mode):
    """
    Feature: prelu_grad ops.
    Description: test ops prelu_grad dynamic tensor input.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    dy_dyn = Tensor(shape=None, dtype=ms.float32)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    weight_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.auto_generate.PReLUGrad())
    test_cell.set_inputs(dy_dyn, x_dyn, weight_dyn)
    dy1 = Tensor(np.array([[1., 1.],
                           [1., 1.],
                           [1., 1.]]).astype(np.float32))
    x1 = Tensor(np.arange(-6, 0).reshape((3, 2)).astype(np.float32))
    weight1 = Tensor(np.array([0.1, 0.6]).astype(np.float32))
    x_output1, weight_ouput1 = test_cell(dy1, x1, weight1)
    expect_x_output1 = np.array([[0.1, 0.6],
                                 [0.1, 0.6],
                                 [0.1, 0.6]]).astype(np.float32)
    expect_weight_ouput1 = np.array([-12., -9.]).astype(np.float32)
    np.testing.assert_array_almost_equal(x_output1.asnumpy(), expect_x_output1, decimal=4)
    np.testing.assert_array_almost_equal(weight_ouput1.asnumpy(), expect_weight_ouput1, decimal=4)
    dy2 = Tensor(np.arange(1, 5).reshape((2, 2)).astype(np.float32))
    x2 = Tensor(np.arange(1, 5).reshape((2, 2)).astype(np.float32))
    weight2 = Tensor(np.array([0.4, -0.5]).astype(np.float32))
    x_output2, weight_ouput2 = test_cell(dy2, x2, weight2)
    expect_x_output2 = np.array([[1, 2],
                                 [3, 4]]).astype(np.float32)
    expect_weight_ouput2 = np.array([0, 0]).astype(np.float32)
    np.testing.assert_array_almost_equal(x_output2.asnumpy(), expect_x_output2, decimal=4)
    np.testing.assert_array_almost_equal(weight_ouput2.asnumpy(), expect_weight_ouput2, decimal=4)
