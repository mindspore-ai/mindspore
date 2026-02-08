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

import mindspore as ms
from mindspore import mint
from mindspore import Tensor, jit
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


@test_utils.run_with_cell
def forward_group_norm_net(input_x, num_group, num_channels, eps=1e-05, affine=True, dtype=None):
    net = mint.nn.GroupNorm(num_group, num_channels, eps=eps, affine=affine, dtype=dtype)
    return net(input_x)


@test_utils.run_with_cell
def forward_group_norm_net_for_dyn(input_x):
    net = mint.nn.GroupNorm(2, 2)
    return net(input_x)


@test_utils.run_with_cell
def grad_group_norm_net(input_x, num_group, num_channels, eps=1e-05, affine=True, dtype=None):
    net = mint.nn.GroupNorm(num_group, num_channels, eps=eps, affine=affine, dtype=dtype)
    return ms.grad(net)(input_x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=3072.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_groupnorm_para_customed_dtype(mode):
    """
    Feature: GroupNorm
    Description: Verify the result of specifying GroupNorm customed para dtype.
    Expectation: success
    """
    input_x = np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                         [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-1.6059085, -0.6882465, -0.6882465, 0.2294155],
                                [-1.1470774, -0.2294155, 0.6882465, 1.6059085],
                                [-0.6882465, 0.6882465, 1.1470774, 1.1470774],
                                [-0.2294155, -0.6882465, 1.6059085, -1.1470774]],
                               [[-0.08812092, 0.8518356, 0.38185734, -1.0280775],
                                [-1.0280775, -0.08812092, 0.38185734, 0.8518356],
                                [1.791792, -0.55809915, -1.4980557, -0.08812092],
                                [0.8518356, -0.08812092, 1.3218138, -1.9680339]]]]).astype(np.float32)
    expect_output_grad = np.array([[[[-3.3379e-06, -1.4603e-06, -1.4603e-06, 5.0664e-07],
                                     [-2.3842e-06, -4.7684e-07, 1.3709e-06, 3.3379e-06],
                                     [-1.4603e-06, 1.3709e-06, 2.4736e-06, 2.4736e-06],
                                     [-4.7684e-07, -1.4603e-06, 3.3379e-06, -2.3842e-06]],
                                    [[-5.3644e-07, 1.0133e-06, 3.5763e-07, -2.0862e-06],
                                     [-2.0862e-06, -5.3644e-07, 3.5763e-07, 1.0133e-06],
                                     [2.5630e-06, -1.3113e-06, -2.9206e-06, -5.3644e-07],
                                     [1.0133e-06, -5.3644e-07, 1.9073e-06, -3.7253e-06]]]])

    num_channel = 2
    num_group = 2
    x = Tensor(input_x, dtype=ms.float32)
    expect_output_shape = (1, 2, 4, 4)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_group_norm_net(x, num_group, num_channel)
        out_grad = grad_group_norm_net(x, num_group, num_channel)
    elif mode == 'KBK':
        output = (jit(forward_group_norm_net, jit_level="O0"))(x, num_group, num_channel)
        out_grad = (jit(grad_group_norm_net, jit_level="O0"))(x, num_group, num_channel)
    else:
        output = (jit(forward_group_norm_net, backend="GE"))(x, num_group, num_channel)
        out_grad = (jit(grad_group_norm_net, backend="GE"))(x, num_group, num_channel)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
def test_mint_group_norm_dyn():
    """
    Feature: Dynamic shape of GroupNorm.
    Description: test GroupNorm with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.array([[[[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
                            [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32))
    in2 = Tensor(np.array([[[[[1.], [3.], [5.]],
                             [[2.], [6.], [8.]],
                             [[3.], [7.], [7.]],
                             [[4.], [8.], [2.]]],
                            [[[5.], [7.], [6.]],
                             [[3.], [5.], [7.]],
                             [[9.], [2.], [5.]],
                             [[7.], [5.], [1.]]]]]).astype(np.float32))
    TEST_OP(forward_group_norm_net_for_dyn, [[in1], [in2]], disable_case=['EmptyTensor', 'ScalarTensor'])
