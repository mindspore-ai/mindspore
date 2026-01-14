# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import mint
from mindspore import Tensor, jit
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@test_utils.run_with_cell
def forward_layer_norm_net(input_x, normal_shape, eps=1e-05, elementwise_affine=True, dtype=None):
    net = mint.nn.LayerNorm(normal_shape, eps=eps, elementwise_affine=elementwise_affine, dtype=dtype)
    return net(input_x)


@test_utils.run_with_cell
def forward_layer_norm_net_for_dyn(input_x):
    net = mint.nn.LayerNorm(4)
    return net(input_x)


@test_utils.run_with_cell
def grad_layer_norm_net(input_x, normal_shape, eps=1e-05, elementwise_affine=True, dtype=None):
    net = mint.nn.LayerNorm(normal_shape, eps=eps, elementwise_affine=elementwise_affine, dtype=dtype)
    return ms.grad(net)(input_x)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=20.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_LayerNorm_para_customed_dtype(mode):
    """
    Feature: LayerNorm
    Description: Verify the result of specifying LayerNorm customed para dtype.
    Expectation: success
    """
    input_x = np.array([[1, 20, 3, 4],
                        [5, 6, 7, 8,],
                        [9, 10, 11, 12]], dtype=np.float32)
    expect_output = np.array([[-0.7913, 1.7144, -0.5275, -0.3956],
                              [-1.3416, -0.4472, 0.4472, 1.3416],
                              [-1.3416, -0.4472, 0.4472, 1.3416]]).astype(np.float32)
    expect_output_grad = np.zeros((3, 4)).astype(np.float32)

    x = Tensor(input_x, dtype=ms.float32)
    normal_shape = 4
    expect_output_shape = (3, 4)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_layer_norm_net(x, normal_shape)
        out_grad = grad_layer_norm_net(x, normal_shape)
    elif mode == 'KBK':
        output = (jit(forward_layer_norm_net, jit_level="O0"))(x, normal_shape)
        out_grad = (jit(grad_layer_norm_net, jit_level="O0"))(x, normal_shape)
    else:
        output = (jit(forward_layer_norm_net, backend="GE"))(x, normal_shape)
        out_grad = (jit(grad_layer_norm_net, backend="GE"))(x, normal_shape)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=20.0)
def test_mint_layer_norm_dyn():
    """
    Feature: Dynamic shape of LayerNorm.
    Description: test LayerNorm with dynamic rank/shape.
    Expectation: success.
    """
    inputnp_x1 = np.random.randn(1, 2, 4).astype(np.float32)
    inputnp_x2 = np.random.randn(1, 4, 2, 4).astype(np.float32)
    input_x1 = Tensor(inputnp_x1, dtype=ms.float32)
    input_x2 = Tensor(inputnp_x2, dtype=ms.float32)
    TEST_OP(forward_layer_norm_net_for_dyn, [[input_x1], [input_x2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])
