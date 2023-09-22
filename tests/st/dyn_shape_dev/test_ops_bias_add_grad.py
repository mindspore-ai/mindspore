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
import mindspore.context as context
from mindspore import Tensor
from mindspore import ops


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_bias_add_grad_2d(data_type):
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    @ms.jit
    def bias_add_grad_forward_func(dout):
        return ops.auto_generate.bias_add_grad(dout, data_format="NCHW")

    dout = np.ones([2, 3]).astype(data_type)
    output = bias_add_grad_forward_func(Tensor(dout))
    expect_output = np.array([2., 2., 2.]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type",
                         [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64])
def test_bias_add_grad_4d(data_type):
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    @ms.jit
    def bias_add_grad_forward_func(dout):
        return ops.auto_generate.bias_add_grad(dout, data_format="NCHW")

    dout = np.ones([2, 3, 4, 4]).astype(data_type)
    output = bias_add_grad_forward_func(Tensor(dout))
    expect_output = np.array([32, 32, 32]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.complex64, np.complex128])
def test_bias_add_grad_5d(data_type):
    """
    Feature: CPU BiasAddGrad.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    @ms.jit
    def bias_add_grad_forward_func(dout):
        return ops.auto_generate.bias_add_grad(dout, data_format="NCHW")

    dout = np.ones([2, 3, 4, 4, 2]).astype(data_type)
    output = bias_add_grad_forward_func(Tensor(dout))
    expect_output = np.array([64., 64., 64.]).astype(data_type)
    assert np.all(output.asnumpy() == expect_output), "bias_add_grad execute failed, please check current code commit"


@pytest.mark.level0
#@pytest.mark.platform_x86_cpu
#@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_bias_add_grad_vmap():
    """
    Feature: bias_add_grad vmap test.
    Description: test the rightness of basic bias_add_grad vmap vmap
    Expectation: use vmap rule's result equal to manually batched.
    """
    @ms.jit
    def bias_add_grad_forward_func(dout):
        return ops.auto_generate.bias_add_grad(dout, data_format="NCHW")

    context.set_context(mode=ms.GRAPH_MODE)
    vmap_bias_add_grad = ops.vmap(bias_add_grad_forward_func, in_axes=(0))
    x = Tensor(np.array([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                         [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]).astype(np.float32))
    output = vmap_bias_add_grad(x)
    expect_out = np.array([[14, 22],
                           [46, 54]]).astype(np.float32)
    assert np.allclose(output.asnumpy(), expect_out)
