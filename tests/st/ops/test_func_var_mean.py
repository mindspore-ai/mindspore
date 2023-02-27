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
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.var_mean(x, axis=0, ddof=True, keepdims=True)


class NetGpu(nn.Cell):
    def construct(self, x):
        return ops.var_mean(x, axis=2, ddof=3, keepdims=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_var_mean(mode):
    """
    Feature: ops.var_mean
    Description: Verify the result of var_mean on Ascend and CPU
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[-4, -6, -5, 8],
                 [3, 2, -7, 0],
                 [7, -4, -3, 8]],
                [[-7, -7, -4, -5],
                 [-6, -7, 6, -2],
                 [-2, -7, 8, -8.]]])
    net = Net()
    output = net(x)
    expect_output = [[[4.49999952e+00, 4.99999970e-01, 4.99999970e-01, 8.45000076e+01],
                      [4.05000038e+01, 4.05000038e+01, 8.45000076e+01, 1.99999988e+00],
                      [4.05000038e+01, 4.49999952e+00, 6.04999962e+01, 1.27999992e+02]]]
    expect_output1 = [[[-5.50000000e+00, -6.50000000e+00, -4.50000000e+00, 1.50000000e+00],
                       [-1.50000000e+00, -2.50000000e+00, -5.00000000e-01, -1.00000000e+00],
                       [2.50000000e+00, -5.50000000e+00, 2.50000000e+00, 0.00000000e+00]]]
    assert np.allclose(output[0].asnumpy(), expect_output)
    assert np.allclose(output[1].asnumpy(), expect_output1)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_var_mean_gpu(mode):
    """
    Feature: ops.var_mean
    Description: Verify the result of var_mean on GPU
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[-4, -6, -5, 8],
                 [3, 2, -7, 0],
                 [7, -4, -3, 8]],
                [[-7, -7, -4, -5],
                 [-6, -7, 6, -2],
                 [-2, -7, 8, -8.]]])
    net = NetGpu()
    output = net(x)
    expect_output = [[[1.28750000e+02],
                      [6.10000000e+01],
                      [1.22000000e+02]],
                     [[6.75000000e+00],
                      [1.04750000e+02],
                      [1.60750000e+02]]]
    expect_output1 = [[[-1.75000000e+00],
                       [-5.00000000e-01],
                       [2.00000000e+00]],
                      [[-5.75000000e+00],
                       [-2.25000000e+00],
                       [-2.25000000e+00]]]
    assert np.allclose(output[0].asnumpy(), expect_output)
    assert np.allclose(output[1].asnumpy(), expect_output1)
