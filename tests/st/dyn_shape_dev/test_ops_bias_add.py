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
from mindspore import ops
import test_utils


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
#@pytest.mark.platform_arm_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
@pytest.mark.env_onecard
def test_bias_add_4d(mode):
    """
    Feature: BiasAdd 4D.
    Description: test BiasAdd with 4D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.auto_generate.bias_add(x, b, data_format="NCHW")

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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_bias_add_2d(mode):
    """
    Feature: BiasAdd 2D.
    Description: test BiasAdd with 2D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.auto_generate.bias_add(x, b, data_format="NCHW")

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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_bias_add_3d(mode):
    """
    Feature: BiasAdd 3D.
    Description: test BiasAdd with 3D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.auto_generate.bias_add(x, b, data_format="NCHW")

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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_bias_add_5d(mode):
    """
    Feature: BiasAdd 5D.
    Description: test BiasAdd with 5D inputs.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.auto_generate.bias_add(x, b, data_format="NCHW")

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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_bias_add_backward(mode):
    """
    Feature: BiasAdd Grad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    @test_utils.run_with_cell
    def bias_add_backward_func(x, b):
        return ops.grad(ops.auto_generate.bias_add, (0,))(x, b, "NCHW")

    ms.context.set_context(mode=mode)
    x = np.ones((2, 3)).astype(np.float32)
    b = np.ones((3,)).astype(np.float32)
    output = bias_add_backward_func(Tensor(x), Tensor(b))
    expect_output = np.ones((2, 3)).astype(np.float32)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE])
def test_bias_add_vmap(mode):
    """
    Feature: biasadd vmap test.
    Description: test the rightness of basic biasadd vmap
    Expectation: use vmap rule's result equal to manually batched.
    """
    @test_utils.run_with_cell
    def bias_add_forward_func(x, b):
        return ops.auto_generate.bias_add(x, b, data_format="NCHW")

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
