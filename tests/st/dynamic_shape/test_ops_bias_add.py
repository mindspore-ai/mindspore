# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, jit
from mindspore.common.api import _pynative_executor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_bias_add_4d(mode):
    """
    Feature: BiasAdd 4D.
    Description: test BiasAdd with 4D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    ms.context.set_context(mode=mode)
    x_shape = [2, 3, 4, 5]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([0.3, 0.5, 0.7]).astype(np.float32)
    output = bias_add_forward_func(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_bias_add_2d(mode):
    """
    Feature: BiasAdd 2D.
    Description: test BiasAdd with 2D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    ms.context.set_context(mode=mode)
    x_shape = [2, 3]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([3, 5, 7]).astype(np.float32)
    output = bias_add_forward_func(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_bias_add_3d(mode):
    """
    Feature: BiasAdd 3D.
    Description: test BiasAdd with 3D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    ms.context.set_context(mode=mode)
    x_shape = [2, 3, 4]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([3, 5, 7]).astype(np.float32)
    output = bias_add_forward_func(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_bias_add_5d(mode):
    """
    Feature: BiasAdd 5D.
    Description: test BiasAdd with 5D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    ms.context.set_context(mode=mode)
    x_shape = [2, 5, 2, 3, 4]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([1, 3, 5, 7, 9]).astype(np.float32)
    output = bias_add_forward_func(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_bias_add_backward(mode):
    """
    Feature: BiasAdd Grad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    @test_utils.run_with_cell
    def bias_add_backward_func(x, b):
        return ops.grad(bias_add_forward_func, (0,))(x, b)

    ms.context.set_context(mode=mode)
    x = np.ones((2, 3)).astype(np.float32)
    b = np.ones((3,)).astype(np.float32)
    output = bias_add_backward_func(Tensor(x), Tensor(b))
    expect_output = np.ones((2, 3)).astype(np.float32)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_windows'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_bias_add_vmap(mode):
    """
    Feature: biasadd vmap test.
    Description: test the rightness of basic biasadd vmap
    Expectation: use vmap rule's result equal to manually batched.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    # must set mode to ms.GRAPH_MODE, or else would trigger pynative procedure and cause precision problem.
    ms.context.set_context(mode=mode)
    vmap_biasadd = ops.vmap(bias_add_forward_func, in_axes=(0, 0))
    x = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                         [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]).astype(np.float16))
    bias = Tensor(np.array([[0, 1], [1, 2]]).astype(np.float16))
    output = vmap_biasadd(x, bias)
    expect_out = np.array([[[[1, 2],
                             [4, 5]],
                            [[5, 6],
                             [8, 9]]],
                           [[[10, 11],
                             [13, 14]],
                            [[14, 15],
                             [17, 18]]]]).astype(np.float16)
    assert np.allclose(output.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
def test_bias_add_ncdhw(mode):
    """
    Feature: BiasAdd with format NCDHW.
    Description: test BiasAdd with NCDHW inputs.
    Expectation: the result match with expected result.
    """
    @jit
    def bias_add(x, b):
        return ops.BiasAdd(data_format="NCDHW")(x, b)

    ms.context.set_context(mode=mode)
    x_shape = [2, 5, 2, 3, 4]
    x = np.ones(x_shape).astype(np.int64)
    b = np.array([1, 3, 5, 7, 9]).astype(np.int64)
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = x
    for i in range(x_shape[0]):
        for j in range(x_shape[1]):
            expect_output[i][j] = x[i][j] + b[j]
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_bias_add_different_input_types(mode):
    """
    Feature: BiasAdd different input types.
    Description: test BiasAdd with different input types.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.BiasAdd(data_format="NCHW")(x, b)

    ms.context.set_context(mode=mode)
    x_shape = [2, 3, 4]
    x = np.ones(x_shape).astype(np.float32)
    b = np.array([3, 5, 7]).astype(np.float16)
    with pytest.raises(TypeError):
        _ = bias_add_forward_func(Tensor(x), Tensor(b))
        _pynative_executor.sync()
