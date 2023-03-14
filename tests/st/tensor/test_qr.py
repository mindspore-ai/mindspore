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

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y):
        output_q, output_r = x.qr(y)
        return output_q, output_r


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_qr(mode):
    """
    Feature: tensor.qr
    Description: Verify the result of tensor.qr
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.asarray([[20, -31, 7], [4, 270, -90], [-8, 17, -32]]), ms.float32)
    y = True
    output_q, output_r = net(x, y)
    expect_output_q = Tensor(np.asarray([[-0.912871, 0.16366126, 0.37400758], [-0.18257418, -0.9830709, -0.01544376],
                                         [0.36514837, -0.08238228, 0.92729706]]), ms.float32)
    expect_output_r = Tensor(np.asarray([[-21.908903, -14.788506, -1.6431675], [0., -271.9031, 92.25824],
                                         [0., 0., -25.665514]]), ms.float32)
    assert np.allclose(output_q.asnumpy(), expect_output_q.asnumpy())
    assert np.allclose(output_r.asnumpy(), expect_output_r.asnumpy())
