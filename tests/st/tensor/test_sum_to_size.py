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

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x):
        return x.sum_to_size((3, 1, 3))


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_sum_to_size(mode):
    """
    Feature: Tensor.sum_to_size
    Description: Verify the result of sum_to_size
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[[24, 20, 39],
                 [79, 67, 43],
                 [62, 0, 95]],
                [[74, 5, 33],
                 [0, 35, 78],
                 [67, 0, 29]],
                [[45, 42, 77],
                 [70, 61, 72],
                 [23, 82, 47]]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[[165, 87, 177]],
                     [[141, 40, 140]],
                     [[138, 185, 196]]]
    assert np.allclose(output.asnumpy(), expect_output)
