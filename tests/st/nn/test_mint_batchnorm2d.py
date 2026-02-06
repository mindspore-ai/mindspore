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


class BatchNorm2dTraining(nn.Cell):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 dtype=None
                 ) -> None:
        super(BatchNorm2dTraining, self).__init__()
        self.net = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                                  track_running_stats=track_running_stats, dtype=dtype)
        self.net.set_train()

    def construct(self, input_x):
        return self.net(input_x)


@test_utils.run_with_cell
def forward_batch_norm_2d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                              track_running_stats=True, dtype=None):
    net = BatchNorm2dTraining(num_features, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track_running_stats, dtype=dtype)
    return net(input_x)


@test_utils.run_with_cell
def forward_batch_norm_2d_for_dyn(input_x):
    net = BatchNorm2dTraining(4)
    return net(input_x)


@test_utils.run_with_cell
def grad_batch_norm_2d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                           track_running_stats=True, dtype=None):
    net = BatchNorm2dTraining(num_features, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track_running_stats, dtype=dtype)
    return ms.grad(net)(input_x)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm2d(mode):
    """
    Feature: BatchNorm2d
    Description: Verify the result of BatchNorm2d.
    Expectation: success
    """

    input_x = np.array(([[[[-1.5354121, -0.52005873, 1.40983502],
                           [-0.68693462, 0.60301964, -0.85031643]],
                          [[0.08652166, -1.07067522, -0.59284334],
                           [0.43919486, 0.49142256, -1.22026105]],
                          [[1.03299872, -1.80427224, 0.05192256],
                           [0.47263551, 1.22857151, 0.25965814]],
                          [[0.00461238, -0.76432777, -0.56504992],
                           [-1.09163674, -0.83346904, -0.79697436]]],
                         [[[0.37364849, 0.01940611, -1.21649304],
                           [-0.06761631, -0.00876229, 1.98813614]],
                          [[0.56870596, 0.86725446, 1.13441053],
                           [-0.36040843, 0.76225533, 0.4763739]],
                          [[0.59319351, -0.44897178, -0.29207948],
                           [-1.34805322, 0.30789763, -0.04003651]],
                          [[-0.10881951, 0.27876277, 0.5238968],
                           [1.52887945, 2.18033473, -1.3915098]]]])).astype(np.float32)
    expect_output = np.array([[[[-1.51146770, -0.48455212, 1.46731818],
                                [-0.65332824, 0.65131533, -0.81857055]],
                               [[-0.06146359, -1.63129938, -0.98307985],
                                [0.41696748, 0.48781881, -1.83422518]],
                               [[1.21342003, -2.12302661, 0.05973814],
                                [0.55446929, 1.44340098, 0.30402169]],
                               [[0.08825184, -0.65838999, -0.46489099],
                                [-0.97620749, -0.72552627, -0.69008988]]],
                              [[[0.41933221, 0.06105589, -1.18891704],
                                [-0.02695750, 0.03256672, 2.05220485]],
                               [[0.59266031, 0.99766660, 1.36008644],
                                [-0.66776216, 0.85522640, 0.46740404]],
                               [[0.69623768, -0.52928114, -0.34478596],
                                [-1.58654261, 0.36074820, -0.04839977]],
                               [[-0.02189066, 0.35445222, 0.59247768],
                                [1.56831706, 2.20088100, -1.26738453]]]]).astype(np.float32)
    expect_output_grad = np.array([[[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                    [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                    [[-1.66688938e-08, 2.91642657e-08, -8.20629786e-10],
                                     [-7.61680941e-09, -1.98281693e-08, -4.17638146e-09]],
                                    [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]],
                                   [[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                    [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
                                    [[-9.56429957e-09, 7.27079685e-09, 4.73636508e-09],
                                     [2.17945217e-08, -4.95564079e-09, 6.64873268e-10]],
                                    [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]]]).astype(np.float32)
    expect_output_shape = (2, 4, 2, 3)
    num_features = 4
    x = Tensor(input_x, dtype=ms.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_batch_norm_2d_net(x, num_features)
        out_grad = grad_batch_norm_2d_net(x, num_features)
    elif mode == 'KBK':
        output = (jit(forward_batch_norm_2d_net, backend="ms_backend", jit_level="O0"))(x, num_features)
        out_grad = (jit(grad_batch_norm_2d_net, backend="ms_backend", jit_level="O0"))(x, num_features)
    else:
        output = (jit(forward_batch_norm_2d_net, backend="GE"))(x, num_features)
        out_grad = (jit(grad_batch_norm_2d_net, backend="GE"))(x, num_features)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(
        output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_batchnorm2d_dyn():
    """
    Feature: Dynamic shape of BatchNorm1d
    Description: test BatchNorm1d with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.random.randn(4, 4, 3, 2), dtype=ms.float32)
    in2 = Tensor(np.random.randn(2, 4, 2, 1), dtype=ms.float32)
    TEST_OP(forward_batch_norm_2d_for_dyn, [[in1], [in2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_resize': True,
                         'disable_tensor_dynamic_type': 'DYNAMIC_RANK'})
