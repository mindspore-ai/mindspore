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
    def construct(self, x):
        out = ops.lp_pool2d(x, norm_type=1, kernel_size=3, stride=1)
        return out


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_lppool1d_normal(mode):
    """
    Feature: LPPool1d
    Description: Verify the result of LPPool1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5)), dtype=ms.float32)
    out = net(x)
    expect_out = np.array([[[[54., 63., 72.],
                             [99., 108., 117.]],
                            [[234., 243., 252.],
                             [279., 288., 297.]],
                            [[414., 423., 432.],
                             [459., 468., 477.]]],
                           [[[594., 603., 612.],
                             [639., 648., 657.]],
                            [[774., 783., 792.],
                             [819., 828., 837.]],
                            [[954., 963., 972.],
                             [999., 1008., 1017.]]]])
    assert np.allclose(out.asnumpy(), expect_out)
