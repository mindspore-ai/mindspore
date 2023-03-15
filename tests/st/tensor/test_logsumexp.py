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
    def construct(self, x, dim):
        return x.logsumexp(dim, keepdims=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logsumexp(mode):
    """
    Feature: tensor.logsumexp
    Description: Verify the result of logsumexp
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.]], ms.float32)
    net = Net()
    output = net(x, dim=0)
    expect_output = [[9.0184765, 10.01848, 11.018479, 12.018477]]
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, dim=1)
    expect_output = [[4.440187],
                     [8.440188],
                     [12.440187]]
    assert np.allclose(output.asnumpy(), expect_output)
