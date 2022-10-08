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


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=3, strides=1)

    def construct(self, x):
        out = self.pool(x)
        return out


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_avgpool3d_normal(mode):
    """
    Feature: AvgPool3d
    Description: Verify the result of AvgPool3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_array = np.arange(1 * 2 * 4 * 4 * 5).reshape((1, 2, 4, 4, 5))
    net = Net()
    x = ms.Tensor(np_array, ms.float32)
    output = net(x)
    expect_output = np.array([[[[[26.0, 27.0, 28.0], [31.0, 32.0, 33.0]], [[46.0, 47.0, 48.0], [51.0, 52.0, 53.0]]],
                               [[[106.0, 107.0, 108.0], [111.0, 112.0, 113.0]],
                                [[126.0, 127.0, 128.0], [131.0, 132.0, 133.0]]]]])
    assert output.shape == (1, 2, 2, 2, 3)
    assert np.allclose(output.asnumpy(), expect_output, rtol=1e-3)
    