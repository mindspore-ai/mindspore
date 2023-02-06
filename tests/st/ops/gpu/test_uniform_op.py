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

import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.random_ops import Uniform
import mindspore.common.dtype as mstype
import numpy as np


class Net(nn.Cell):
    def __init__(self, min_val=0.0, max_val=1.0):
        super(Net, self).__init__()
        self.uniform = Uniform(minval=min_val, maxval=max_val)

    def construct(self, x):
        return self.uniform(x)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_double():
    """
    Feature: Uniform gpu TEST.
    Description: 2d test case for Uniform
    Expectation: success
    """
    x = Tensor(np.random.randn(3, 4), mstype.float64)
    net = Net(min_val=1.0, max_val=2.0)
    y = net(x)
    assert y.shape == (3, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_float():
    """
    Feature: Uniform gpu TEST.
    Description: 2d test case for Uniform
    Expectation: success
    """
    x = Tensor(np.random.randn(3, 4), mstype.float32)
    net = Net(min_val=1.0, max_val=2.0)
    y = net(x)
    assert y.shape == (3, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_half():
    """
    Feature: Uniform gpu TEST.
    Description: 2d test case for Uniform
    Expectation: success
    """
    x = Tensor(np.random.randn(3, 4), mstype.float16)
    net = Net(min_val=1.0, max_val=2.0)
    y = net(x)
    assert y.shape == (3, 4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_uniform_int32_input_error():
    """
    Feature: Uniform gpu TEST.
    Description: 2d test case for Uniform
    Expectation: default
    """
    x = Tensor(np.random.randn(3, 4), mstype.int32)
    net = Net(min_val=1.0, max_val=2.0)
    with pytest.raises(TypeError):
        net(x)
