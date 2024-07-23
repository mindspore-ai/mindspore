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

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def sin_forward_func(x):
    return ops.sin(x)


@test_utils.run_with_cell
def sin_backward_func(x):
    return ops.grad(sin_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sin_forward(mode):
    """
    Feature: sin ops.
    Description: test ops sin.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sin_forward_func(x)
    expect_output = np.asarray([0.5810352, 0.27635565, 0.41687083, 0.5810352]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sin_backward(mode):
    """
    Feature: sin ops.
    Description: test auto grad of ops sin.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sin_backward_func(x)
    expect_output = np.asarray([0.8138785, 0.96105546, 0.90896577, 0.8138785]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sin_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sin ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[0.62, 0.28, 0.43, 0.62]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(sin_forward_func))
    output = nest_vmap(x)
    expect_out = sin_forward_func(x)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sin_dynamic(mode):
    """
    Feature: sin ops.
    Description: test ops sin dynamic tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.sin)
    test_cell.set_inputs(x_dyn)
    x1 = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output1 = test_cell(x1)
    expect_output1 = np.asarray([0.5810352, 0.27635565, 0.41687083, 0.5810352]).astype(np.float32)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect_output1, decimal=4)
    x2 = Tensor(np.array([[0.62, 0.28],
                          [0.43, 0.62]]).astype(np.float32))
    output2 = test_cell(x2)
    expect_output2 = np.asarray([[0.5810352, 0.27635565],
                                 [0.41687083, 0.5810352]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect_output2, decimal=4)
