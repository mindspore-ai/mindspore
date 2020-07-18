# Copyright 2020 Huawei Technologies Co., Ltd
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
"""ut for batchnorm layer"""
import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor, Parameter


def test_bn_pars_valid1():
    """ut of BatchNorm parameters' validation"""
    with pytest.raises(ValueError):
        nn.BatchNorm2d(num_features=0)


def test_bn_pars_valid2():
    """ut of BatchNorm parameters' validation"""
    with pytest.raises(ValueError):
        nn.BatchNorm2d(num_features=3, momentum=-0.1)


def test_bn_init():
    """ut of BatchNorm parameters' validation"""
    bn = nn.BatchNorm2d(num_features=3)

    assert isinstance(bn.gamma, Parameter)
    assert isinstance(bn.beta, Parameter)
    assert isinstance(bn.moving_mean, Parameter)
    assert isinstance(bn.moving_variance, Parameter)


def test_bn2d():
    """ut of nn.BatchNorm2d"""
    gamma = Tensor(np.array([0.1, 0.3, 0.4]).astype(np.float32))
    beta = Tensor(np.zeros((3), dtype=np.float32))
    moving_mean = Tensor(np.zeros((3), dtype=np.float32))
    moving_var = Tensor(np.ones((3), dtype=np.float32))

    bn = nn.BatchNorm2d(num_features=3,
                        eps=1e-5,
                        momentum=0.1,
                        gamma_init=gamma,
                        beta_init=beta,
                        moving_mean_init=moving_mean,
                        moving_var_init=moving_var)

    # 3-channel RGB
    input_data = Tensor(np.random.randint(0, 1, [1, 3, 224, 224]).astype(np.float32))
    output = bn(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0][0][0], (np.float32, np.float64))


def test_bn1d():
    """ut of nn.BatchNorm1d"""
    bn = nn.BatchNorm1d(3)
    input_data = Tensor(np.random.randint(0, 1, [1, 3]).astype(np.float32))
    output = bn(input_data)
    output_np = output.asnumpy()
    assert isinstance(output_np[0][0], (np.float32, np.float64))


def test_bn2d_train():
    """ut of nn.BatchNorm2d training"""
    bn = nn.BatchNorm2d(3)
    bn.training = True
    Tensor(np.random.randint(0, 255, [1, 3, 224, 224]))
    # current operator does not support multiple output
