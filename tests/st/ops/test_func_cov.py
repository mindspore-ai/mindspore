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
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x, correction=1, fweights=None, aweights=None):
        output = ops.cov(x, correction=correction, fweights=fweights, aweights=aweights)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cov_normal(mode):
    """
    Feature: cov
    Description: Verify the result of cov
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[0., 2.], [1., 1.], [2., 0.]]).T
    output1 = net(x)
    expect_output1 = np.array([[1., -1.],
                               [-1., 1.]])
    output2 = net(x, correction=0)
    expect_output2 = np.array([[0.6666667, -0.6666667],
                               [-0.6666667, 0.6666667]])
    fw = ms.Tensor([5, 2, 4], dtype=ms.int64)
    aw = ms.Tensor([0.4588, 0.9083, 0.7616], ms.float32)
    output3 = net(x, fweights=fw, aweights=aw)
    expect_output3 = np.array([[0.81504613, -0.81504613],
                               [-0.81504613, 0.81504613]])
    assert np.allclose(output1.asnumpy(), expect_output1)
    assert np.allclose(output2.asnumpy(), expect_output2)
    assert np.allclose(output3.asnumpy(), expect_output3)
