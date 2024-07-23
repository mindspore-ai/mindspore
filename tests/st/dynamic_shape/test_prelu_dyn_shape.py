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
def prelu_forward_func(x, weight):
    return ops.prelu(x, weight)


@test_utils.run_with_cell
def prelu_backward_func(x, weight):
    return ops.grad(prelu_forward_func, (0, 1))(x, weight)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_prelu_forward(mode):
    """
    Feature: prelu ops.
    Description: test ops prelu.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output = prelu_forward_func(x, weight)
    expect_output = np.array([[[-0.6, -0.5],
                               [-2.4, -1.8],
                               [0.6, 0.3]],
                              [[0., 1.],
                               [2., 3.],
                               [4., 5.]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_prelu_backward(mode):
    """
    Feature: prelu ops.
    Description: test auto grad of ops prelu.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output = prelu_backward_func(x, weight)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_prelu_dynamic(mode):
    """
    Feature: prelu ops.
    Description: test ops prelu dynamic tensor input.
    Expectation: output right results.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    weight_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.prelu)
    test_cell.set_inputs(x_dyn, weight_dyn)
    x1 = Tensor(np.arange(-6, 6).reshape((2, 3, 2)).astype(np.float32))
    weight1 = Tensor(np.array([0.1, 0.6, -0.3]).astype(np.float32))
    output1 = test_cell(x1, weight1)
    expect_output1 = np.array([[[-0.6, -0.5],
                                [-2.4, -1.8],
                                [0.6, 0.3]],
                               [[0., 1.],
                                [2., 3.],
                                [4., 5.]]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect_output1, decimal=4)
    x2 = Tensor(np.arange(-4, 0).reshape((2, 2)).astype(np.float32))
    weight2 = Tensor(np.array([0.4, -0.5]).astype(np.float32))
    output2 = test_cell(x2, weight2)
    expect_output2 = np.array([[-1.6, 1.5],
                               [-0.8, 0.5]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect_output2, decimal=4)
