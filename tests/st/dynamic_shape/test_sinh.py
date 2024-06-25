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
def sinh_forward_func(x):
    return ops.sinh(x)


@test_utils.run_with_cell
def sinh_backward_func(x):
    return ops.grad(sinh_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sinh_forward(mode):
    """
    Feature: sinh ops.
    Description: test ops sinh.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sinh_forward_func(x)
    expect_output = np.asarray([0.6604918, 0.28367308, 0.44337422, 0.6604918]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sinh_backward(mode):
    """
    Feature: sinh ops.
    Description: test auto grad of ops sinh.
    Expectation: output the right grad.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output = sinh_backward_func(x)
    expect_output = np.asarray([1.1984363, 1.0394568, 1.0938833, 1.1984363]).astype(np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expect_output, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sinh_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sinh ops vmap.
    Expectation: expect right result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[[0.62, 0.28, 0.43, 0.62]]]).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(sinh_forward_func))
    output = nest_vmap(x)
    expect_out = sinh_forward_func(x)
    np.testing.assert_equal(output.asnumpy(), expect_out.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sinh_dynamic(mode):
    """
    Feature: sinh ops.
    Description: test ops sinh dynamic tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(ops.sinh)
    test_cell.set_inputs(x_dyn)
    x1 = Tensor(np.array([0.62, 0.28, 0.43, 0.62]).astype(np.float32))
    output1 = test_cell(x1)
    expect_output1 = np.asarray([0.6604918, 0.28367308, 0.44337422, 0.6604918]).astype(np.float32)
    np.testing.assert_array_almost_equal(output1.asnumpy(), expect_output1, decimal=4)
    x2 = Tensor(np.array([[0.62, 0.28],
                          [0.43, 0.62]]).astype(np.float32))
    output2 = test_cell(x2)
    expect_output2 = np.asarray([[0.6604918, 0.28367308],
                                 [0.44337422, 0.6604918]]).astype(np.float32)
    np.testing.assert_array_almost_equal(output2.asnumpy(), expect_output2, decimal=4)
