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
import mindspore.mint.nn as nn
from mindspore import Tensor, jit
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


class BatchNorm1dTraining(nn.Cell):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 dtype=None
                 ) -> None:
        super(BatchNorm1dTraining, self).__init__()
        self.net = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine,
                                  track_running_stats=track_running_stats, dtype=dtype)
        self.net.set_train()

    def construct(self, input_x):
        return self.net(input_x)


@test_utils.run_with_cell
def forward_batch_norm_1d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                              track_running_stats=True, dtype=None):
    net = BatchNorm1dTraining(num_features, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track_running_stats, dtype=dtype)
    return net(input_x)


@test_utils.run_with_cell
def forward_batch_norm_1d_for_dyn(input_x):
    net = BatchNorm1dTraining(4)
    return net(input_x)


@test_utils.run_with_cell
def grad_batch_norm_1d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                           track_running_stats=True, dtype=None):
    net = BatchNorm1dTraining(num_features, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track_running_stats, dtype=dtype)
    return ms.grad(net)(input_x)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm1d(mode):
    """
    Feature: BatchNorm1d
    Description: Verify the result of BatchNorm1d.
    Expectation: success
    """

    input_x = np.array([[0.7, 0.5, 0.5, 0.6],
                        [0.5, 0.4, 0.6, 0.9]]).astype(np.float32)
    expect_output = np.array([[0.99950004, 0.99800587, -0.99800587, -0.99977762],
                              [-0.99950075, -0.99800611, 0.99800587, 0.99977809]]).astype(np.float32)
    expect_output_grad = np.array([[2.97576730e-06, -5.92488232e-06, 0.00000000e+00, 0.00000000e+00],
                                   [-2.97576889e-06, 5.92487913e-06, 0.00000000e+00, 0.00000000e+00]]).astype(
                                       np.float32)
    expect_output_shape = (2, 4)
    num_features = 4
    x = Tensor(input_x, dtype=ms.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_batch_norm_1d_net(x, num_features)
        out_grad = grad_batch_norm_1d_net(x, num_features)
    elif mode == 'KBK':
        output = (jit(forward_batch_norm_1d_net, backend="ms_backend", jit_level="O0"))(x, num_features)
        out_grad = (jit(grad_batch_norm_1d_net, backend="ms_backend", jit_level="O0"))(x, num_features)
    else:
        output = (jit(forward_batch_norm_1d_net, backend="GE"))(x, num_features)
        out_grad = (jit(grad_batch_norm_1d_net, backend="GE"))(x, num_features)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(
        output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_batchnorm1d_dyn():
    """
    Feature: Dynamic shape of BatchNorm1d
    Description: test BatchNorm1d with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.array([[1, 3, 3, 5], [2, 4, 6, 8]]), dtype=ms.float32)
    in2 = Tensor(
        np.array([[1, 3, 3, 5], [2, 4, 6, 8], [2, 5, 1, 8]]), dtype=ms.float32)
    TEST_OP(forward_batch_norm_1d_for_dyn, [[in1], [in2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_resize': True,
                         'disable_tensor_dynamic_type': 'DYNAMIC_RANK'})
