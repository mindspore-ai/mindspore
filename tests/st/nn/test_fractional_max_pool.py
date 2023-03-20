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

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore as ms


class FractionalMaxPool2dNet(nn.Cell):
    """FractionalMaxPool2d"""

    def __init__(self):
        super(FractionalMaxPool2dNet, self).__init__()
        _random_samples = Tensor(np.array([[[0.0, 0.0]]]), mstype.float32)
        self.pool1 = nn.FractionalMaxPool2d(kernel_size=2, output_size=(2, 2), _random_samples=_random_samples,
                                            return_indices=True)
        self.pool2 = nn.FractionalMaxPool2d(kernel_size=2, output_ratio=(0.5, 0.5), _random_samples=_random_samples,
                                            return_indices=True)

    def construct(self, x):
        output1 = self.pool1(x)
        output2 = self.pool2(x)
        return output1, output2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fractional_maxpool2d_normal(mode):
    """
    Feature: FractionalMaxPool2d
    Description: Verify the result of FractionalMaxPool2d
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = FractionalMaxPool2dNet()
    input_x = Tensor(np.random.rand(25).reshape([1, 1, 5, 5]), mstype.float32)
    output1, output2 = net(input_x)
    assert output1[0].shape == output1[1].shape == (1, 1, 2, 2)
    assert output2[0].shape == output2[1].shape == (1, 1, 2, 2)
    input_x = Tensor([[[[5.58954370e-001, 6.63938331e-001, 6.21228504e-001, 2.42979444e-001, 3.76893662e-001],
                        [1.81983045e-003, 3.52343421e-001, 4.62048613e-001, 1.10343760e-001, 1.39571702e-001],
                        [4.99799584e-001, 4.64907907e-001, 6.20357162e-001, 3.59420753e-001, 1.26215309e-001],
                        [7.71829579e-002, 4.58553624e-001, 3.58015698e-001, 3.53923170e-001, 1.75972716e-001],
                        [5.65106732e-001, 6.46603699e-001, 6.05013040e-001, 3.82114821e-001, 4.62306777e-003]]]],
                     mstype.float32)
    output1, output2 = net(input_x)
    expect_output_y = np.array([[[[6.63938344e-001, 3.76893669e-001],
                                  [6.46603703e-001, 3.82114828e-001]]]])
    expect_output_argmax = np.array([[[[1, 4],
                                       [21, 23]]]])
    assert np.allclose(output1[0].asnumpy(), expect_output_y)
    assert np.allclose(output1[1].asnumpy(), expect_output_argmax)
    assert np.allclose(output2[0].asnumpy(), expect_output_y)
    assert np.allclose(output2[1].asnumpy(), expect_output_argmax)


class FractionalMaxPool3dNet(nn.Cell):
    """FractionalMaxPool3d"""

    def __init__(self):
        super(FractionalMaxPool3dNet, self).__init__()
        _random_samples = Tensor(np.array([0.0, 0.0, 0.0]).reshape([1, 1, 3]), mstype.float32)
        self.pool1 = nn.FractionalMaxPool3d(kernel_size=(1, 1, 1), _random_samples=_random_samples,
                                            output_size=(1, 1, 2), return_indices=True)
        self.pool2 = nn.FractionalMaxPool3d(kernel_size=(1, 1, 1), output_ratio=(0.5, 0.5, 0.5),
                                            _random_samples=_random_samples, return_indices=True)

    def construct(self, x):
        output1 = self.pool1(x)
        output2 = self.pool2(x)
        return output1, output2


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fractional_maxpool3d_normal(mode):
    """
    Feature: Test FractioanlMaxPool3d
    Description: Test the functionality of FractionalMaxPool3d
    Expectation: Success
    """
    ms.set_context(mode=mode)
    input_x = Tensor(np.random.rand(16).reshape([1, 1, 2, 2, 4]), mstype.float32)
    net = FractionalMaxPool3dNet()
    output1, output2 = net(input_x)
    assert output1[0].shape == output1[1].shape == (1, 1, 1, 1, 2)
    assert output2[0].shape == output2[1].shape == (1, 1, 1, 1, 2)
    input_x = Tensor([[[[[5.76273143e-001, 7.97047436e-001, 5.05385816e-001, 7.98332036e-001],
                         [5.79880655e-001, 9.75979388e-001, 3.17571498e-002, 8.08261558e-002]],
                        [[3.82758647e-001, 7.09801614e-001, 4.39641386e-001, 5.71077049e-001],
                         [9.16305065e-001, 3.71438652e-001, 6.52868748e-001, 6.91260636e-001]]]]], mstype.float32)
    output1, output2 = net(input_x)
    expect_output_y = np.array([[[[[9.16305065e-001, 6.91260636e-001]]]]])
    expect_output_argmax = np.array([[[[[12, 15]]]]])
    assert np.allclose(output1[0].asnumpy(), expect_output_y)
    assert np.allclose(output1[1].asnumpy(), expect_output_argmax)
    assert np.allclose(output2[0].asnumpy(), expect_output_y)
    assert np.allclose(output2[1].asnumpy(), expect_output_argmax)
