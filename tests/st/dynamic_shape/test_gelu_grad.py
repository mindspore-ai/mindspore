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
from mindspore import Tensor
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def gelu_grad_forward_func(dy, x, y):
    return ops.auto_generate.GeLUGrad()(dy, x, y)


@test_utils.run_with_cell
def gelu_grad_backward_func(dy, x, y):
    return ops.grad(gelu_grad_forward_func, (0,))(dy, x, y)


@test_utils.run_with_cell
def gelu_grad_dyn_shape_func(dy, x, y):
    return ops.auto_generate.GeLUGrad()(dy, x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_gelu_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op gelu_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy = Tensor(np.array([1.0829641, 1.0860993, 1.0115843]).astype('float32'))
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype('float32'))
    y = Tensor(np.array([1.0, 2.0, 3.0]).astype('float32'))
    out = gelu_grad_forward_func(dy, x, y)
    expect = np.array([1.1728112, 1.1796116, 1.0233028]).astype('float32')
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_gelu_grad_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of avg pool.
    Description: test dynamic tensor and dynamic scalar of avg pool.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    x_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    y_dyn = ms.Tensor(shape=[None], dtype=ms.float32)
    dy = Tensor(np.array([1.0829641, 1.0860993, 1.0115843]).astype('float32'))
    x = Tensor(np.array([1.0, 2.0, 3.0]).astype('float32'))
    y = Tensor(np.array([1.0, 2.0, 3.0]).astype('float32'))
    test_cell = test_utils.to_cell_obj(gelu_grad_dyn_shape_func)
    test_cell.set_inputs(dy_dyn, x_dyn, y_dyn)
    out = test_cell(dy, x, y)
    expect = np.array([1.1728112, 1.1796116, 1.0233028]).astype('float32')
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    dy1 = Tensor(
        np.array([1.0829641, 1.0860993, 1.0115843, 1.0]).astype('float32'))
    x1 = Tensor(np.array([1.0, 2.0, 3.0, 4.0]).astype('float32'))
    y1 = Tensor(np.array([1.0, 2.0, 3.0, 4.0]).astype('float32'))
    output1 = test_cell(dy1, x1, y1)
    expect1 = np.array(
        [1.172811, 1.1796113, 1.0233024, 1.000335]).astype('float32')
    assert np.allclose(output1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_gelu_grad_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of gelu.
    Description: test dynamic rank tensor of gelu.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    dy_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(gelu_grad_dyn_shape_func)
    dy = Tensor(
        np.array([[1.0829641, 1.0860993, 1.0115843]]).astype('float32'))
    x = Tensor(np.array([[1.0, 2.0, 3.0]]).astype('float32'))
    y = Tensor(np.array([[1.0, 2.0, 3.0]]).astype('float32'))
    test_cell.set_inputs(dy_dyn, x_dyn, y_dyn)
    output = test_cell(dy, x, y)
    expect = np.array([[1.1728112, 1.1796116, 1.0233028]]).astype('float32')
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    dy1 = Tensor(
        np.array([[1.0829641, 1.0860993, 1.0115843, 1.0]]).astype('float32'))
    x1 = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]).astype('float32'))
    y1 = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]).astype('float32'))
    output1 = test_cell(dy1, x1, y1)
    expect1 = np.array(
        [[1.172811, 1.1796113, 1.0233024, 1.000335]]).astype('float32')
    assert np.allclose(output1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)
