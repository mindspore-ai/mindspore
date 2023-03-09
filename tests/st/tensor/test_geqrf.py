# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore import nn, Tensor
import mindspore as ms


RTOL = 1.e-5
ATOL = 1.e-6


class GeqrfNet(nn.Cell):
    def construct(self, x):
        return x.geqrf()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_geqrf_rank2_double_fp(mode):
    """
    Feature: ops.geqrf
    Description: Verify the result of ops.geqrf
    Expectation: success
    """
    x_np = np.array([[15.5862, 10.6579],
                     [0.1885, -10.0553],
                     [4.4496, 0.7312]]).astype(np.float64)
    expect_y = np.array([[-1.62100001e+01, -1.03315412e+01],
                         [5.92838136e-03, 1.04160357e+01],
                         [1.39941250e-01, 1.07113682e-01]])
    expect_tau = np.array([1.96151758, 1.97731361])

    net = GeqrfNet()
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_geqrf_rank3_float_fp(mode):
    """
    Feature: ops.geqrf
    Description: Verify the result of ops.geqrf
    Expectation: success
    """
    x_np = np.array([[[1.7705, 0.5642],
                      [0.8125, 0.4068],
                      [0.1714, -0.1203]],

                     [[-1.5411, 1.5866],
                      [-0.3722, -0.9087],
                      [-0.0862, -0.7602]]]).astype(np.float32)
    expect_y = np.array([[[-1.9555572, -0.6692831],
                          [0.2180589, -0.2243657],
                          [0.0460004, -0.4888012]],

                         [[1.5877508, -1.2856942],
                          [0.1189574, 1.5059648],
                          [0.0275500, 0.3045090]]]).astype(np.float32)
    expect_tau = np.array([[1.9053684, 1.6143007], [1.9706184, 1.8302855]]).astype(np.float32)
    net = GeqrfNet()
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)
