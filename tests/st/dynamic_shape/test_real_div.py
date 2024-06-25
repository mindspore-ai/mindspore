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
def real_div_forward_func(x, y):
    return ops.RealDiv()(x, y)


@test_utils.run_with_cell
def real_div_backward_func(x, y):
    return ops.grad(real_div_forward_func, (0, 1))(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_real_div_forward(mode):
    """
    Feature: real_div ops.
    Description: test ops real_div.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
    output = real_div_forward_func(x, y)
    expect_output = np.asarray([0.25, 0.4, 0.5]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_real_div_backward(mode):
    """
    Feature: real_div ops.
    Description: test auto grad of ops real_div.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
    dx, dy = real_div_backward_func(x, y)
    except_dx = np.asarray([0.25, 0.2, 0.1666667]).astype(np.float32)
    except_dy = np.asarray([-0.0625, -0.08, -0.0833333]).astype(np.float32)
    np.testing.assert_array_almost_equal(dx.asnumpy(), except_dx, decimal=4)
    np.testing.assert_array_almost_equal(dy.asnumpy(), except_dy, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_real_div_vmap(mode):
    """
    Feature: test vmap function.
    Description: test real_div ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[1.0, 2.0, 3.0]]]).astype(np.float32))
    y = Tensor(np.array([[[4.0, 5.0, 6.0]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(real_div_forward_func, in_axes=(0, 0)), in_axes=(0, 0))
    output = nest_vmap(x, y)
    expect_out = real_div_forward_func(x, y)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_real_div_dynamic(mode):
    """
    Feature: real_div ops.
    Description: test ops real_div dynamic tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x_dyn = Tensor(shape=None, dtype=ms.float32)
    y_dyn = Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.RealDiv())
    test_cell.set_inputs(x_dyn, y_dyn)
    x1 = Tensor(np.array([1.0, 2.0, 3.0]).astype(np.float32))
    y1 = Tensor(np.array([4.0, 5.0, 6.0]).astype(np.float32))
    output1 = test_cell(x1, y1)
    expect_output1 = np.asarray([0.25, 0.4, 0.5]).astype(np.float32)
    np.testing.assert_allclose(output1.asnumpy(), expect_output1)
    x2 = Tensor(np.array([[1.0, 2.0],
                          [3.0, 4.0]]).astype(np.float32))
    y2 = Tensor(np.array(2.0).astype(np.float32))
    output2 = test_cell(x2, y2)
    expect_output2 = np.asarray([[0.5, 1.0],
                                 [1.5, 2.0]]).astype(np.float32)
    np.testing.assert_allclose(output2.asnumpy(), expect_output2)
