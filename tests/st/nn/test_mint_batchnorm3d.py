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


class BatchNorm3dTraining(nn.Cell):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,
                 dtype=None
                 ) -> None:
        super(BatchNorm3dTraining, self).__init__()
        self.net = nn.BatchNorm3d(num_features, eps=eps, momentum=momentum, affine=affine,
                                  track_running_stats=track_running_stats, dtype=dtype)
        self.net.set_train()

    def construct(self, input_x):
        return self.net(input_x)


@test_utils.run_with_cell
def forward_batch_norm_3d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                              track_running_stats=True, dtype=None):
    net = BatchNorm3dTraining(num_features, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track_running_stats, dtype=dtype)
    return net(input_x)


@test_utils.run_with_cell
def forward_batch_norm_3d_for_dyn(input_x):
    net = BatchNorm3dTraining(4)
    return net(input_x)


@test_utils.run_with_cell
def grad_batch_norm_3d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                           track_running_stats=True, dtype=None):
    net = BatchNorm3dTraining(num_features, eps=eps, momentum=momentum, affine=affine,
                              track_running_stats=track_running_stats, dtype=dtype)
    return ms.grad(net)(input_x)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_batchnorm3d(mode):
    """
    Feature: BatchNorm3d
    Description: Verify the result of BatchNorm3d.
    Expectation: success
    """

    input_x = np.array(
        [[[[[-0.19553653, -0.38225709, 0.65811716],
            [0.82997666, -0.82071, 1.52072142]],
           [[0.62595402, -2.08246214, 0.79570662],
            [0.24145219, 0.06299961, 0.77957082]]],
          [[[-1.23532939, -0.60160848, -1.33098584],
            [-0.11987434, 0.54427765, -1.66259654]],
           [[0.42377281, -0.82846511, -0.17797144],
            [0.73298768, 1.02121543, 0.81248093]]],
          [[[1.28061095, -0.59624903, -0.1114635],
            [-1.71278863, -1.17056539, -1.49965787]],
           [[-0.4567144, -0.24608901, -0.79492688],
            [-0.49285035, 0.78368926, 1.89284614]]],
          [[[0.82389551, -0.02417875, 0.1353407],
            [-0.26898776, 0.54589644, 0.97872609]],
           [[1.66718225, 0.65020981, -2.46485396],
            [0.47076539, -0.38466012, -0.76017557]]]],
         [[[[-1.43258207, 1.4440276, -1.16539963],
            [0.30640335, 0.93569623, -0.23791705]],
           [[0.1870397, -0.67716751, 1.14585376],
            [-0.50472719, 0.35820541, 0.47061354]]],
          [[[0.17213828, 0.88083509, -0.33014235],
            [0.28616101, -0.56810896, -0.43235523]],
           [[-2.01753406, 0.35113867, 1.14428865],
            [-1.14061447, -0.33881645, 1.63555686]]],
          [[[-1.03782811, -1.43242332, -0.93256603],
            [-0.52446202, -1.30753501, 0.74597684]],
           [[-0.25775732, 0.43614491, -0.54276708],
            [0.29599765, 0.48494346, 0.9820326]]],
          [[[-1.26525238, -0.93627899, -0.05923393],
            [1.52983383, 0.11205568, 0.18655583]],
           [[0.96446208, -1.27806913, -0.01856735],
            [-0.73483061, 0.80634165, 0.2402837]]]]]
    ).astype(np.float32)
    expect_output = np.array(
        [[[[[-3.54653776e-01, -5.64978302e-01, 6.06912971e-01],
            [8.00497770e-01, -1.05885732e+00, 1.57856166e+00]],
           [[5.70683956e-01, -2.48011136e+00, 7.61895537e-01],
            [1.37576044e-01, -6.34352714e-02, 7.43719935e-01]]],
          [[[-1.20689416e+00, -5.23710608e-01, -1.31001663e+00],
            [-4.37664567e-03, 7.11613178e-01, -1.66751003e+00]],
           [[5.81702769e-01, -7.68273711e-01, -6.70082793e-02],
            [9.15052235e-01, 1.22577643e+00, 1.00074995e+00]]],
          [[[1.65888464e+00, -3.63463193e-01, 1.58901304e-01],
            [-1.56655324e+00, -9.82298613e-01, -1.33690131e+00]],
           [[-2.13112295e-01, 1.38400672e-02, -5.77541888e-01],
            [-2.52049416e-01, 1.12344337e+00, 2.31857824e+00]]],
          [[[8.42025101e-01, -6.68345541e-02, 1.04118399e-01],
            [-3.29190165e-01, 5.44100523e-01, 1.00795305e+00]],
           [[1.74575400e+00, 6.55890524e-01, -2.68244410e+00],
            [4.63584483e-01, -4.53153282e-01, -8.55583668e-01]]]],
         [[[[-1.74807799e+00, 1.49217272e+00, -1.44712031e+00],
            [2.10737869e-01, 9.19581652e-01, -4.02391762e-01]],
           [[7.62851089e-02, -8.97169232e-01, 1.15630579e+00],
            [-7.02930152e-01, 2.69088417e-01, 3.95706385e-01]]],
          [[[3.10427874e-01, 1.07443929e+00, -2.31056303e-01],
            [4.33350205e-01, -4.87596482e-01, -3.41246992e-01]],
           [[-2.05015063e+00, 5.03399432e-01, 1.35845566e+00],
            [-1.10478675e+00, -2.40407437e-01, 1.88806784e+00]]],
          [[[-8.39271963e-01, -1.26445496e+00, -7.25850403e-01],
            [-2.86111504e-01, -1.12988579e+00, 1.08280754e+00]],
           [[1.26728509e-03, 7.48958528e-01, -3.05835515e-01],
            [5.97947478e-01, 8.01539719e-01, 1.33716154e+00]]],
          [[[-1.39686155e+00, -1.04430926e+00, -1.04402296e-01],
            [1.59856117e+00, 7.91644305e-02, 1.59004346e-01]],
           [[9.92666721e-01, -1.41059697e+00, -6.08209558e-02],
            [-8.28422070e-01, 8.23213041e-01, 2.16583133e-01]]]]]
    ).astype(np.float32)
    expect_output_grad = np.array(
        [[[[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
           [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]],
          [[[2.09010445e-08, 9.06964281e-09, 2.26869226e-08],
            [7.57949536e-11, -1.23237474e-08, 2.88780093e-08]],
           [[-1.00739532e-08, 1.33049953e-08, 1.16045218e-09],
            [-1.58469131e-08, -2.12280487e-08, -1.73310308e-08]]],
          [[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
           [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]],
          [[[-4.80341100e-09, 3.81263965e-10, -5.93953109e-10],
            [1.87789606e-09, -3.10387249e-09, -5.74996317e-09]],
           [[-9.95881688e-09, -3.74158882e-09, 1.53022537e-08],
            [-2.64456124e-09, 2.58505528e-09, 4.88075713e-09]]]],
         [[[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
           [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]],
          [[[-5.37600320e-09, -1.86071851e-08, 4.00144273e-09],
            [-7.50477724e-09, 8.44421599e-09, 5.90972959e-09]],
           [[3.55045948e-08, -8.71789307e-09, -2.35257929e-08],
            [1.91327416e-08, 4.16338608e-09, -3.26976384e-08]]],
          [[[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]],
           [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]]],
          [[[7.96852717e-09, 5.95735949e-09, 5.95572647e-10],
            [-9.11914277e-09, -4.51600896e-10, -9.07055264e-10]],
           [[-5.66276048e-09, 8.04688227e-09, 3.46958851e-10],
            [4.72581130e-09, -4.69609640e-09, -1.23551891e-09]]]]]
    ).astype(np.float32)
    expect_output_shape = (2, 4, 2, 2, 3)
    num_features = 4
    x = Tensor(input_x, dtype=ms.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_batch_norm_3d_net(x, num_features)
        out_grad = grad_batch_norm_3d_net(x, num_features)
    elif mode == 'KBK':
        output = (jit(forward_batch_norm_3d_net, backend="ms_backend", jit_level="O0"))(x, num_features)
        out_grad = (jit(grad_batch_norm_3d_net, backend="ms_backend", jit_level="O0"))(
            x, num_features)
    else:
        output = (jit(forward_batch_norm_3d_net, backend="GE"))(x, num_features)
        out_grad = (jit(grad_batch_norm_3d_net, backend="GE"))(
            x, num_features)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(
        output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
def test_batchnorm3d_dyn():
    """
    Feature: Dynamic shape of BatchNorm1d
    Description: test BatchNorm1d with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.random.randn(2, 4, 3, 2, 2), dtype=ms.float32)
    in2 = Tensor(np.random.randn(3, 4, 1, 2, 4), dtype=ms.float32)
    TEST_OP(forward_batch_norm_3d_for_dyn, [[in1], [in2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor', 'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'disable_resize': True,
                         'disable_tensor_dynamic_type': 'DYNAMIC_RANK'})
