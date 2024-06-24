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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x, target):
        output0 = ops.huber_loss(x, target, reduction="none", delta=0.5)
        output1 = ops.huber_loss(x, target, reduction="mean", delta=0.5)
        output2 = ops.huber_loss(x, target, reduction="sum", delta=0.5)
        return output0, output1, output2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_huber_loss(mode):
    """
    Feature: Test huber_loss
    Description: Test the functionality of huber_loss
    Expectation: Success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[0.44168271, 0.222135, 0.94593939, 0.90117212],
                    [0.76260363, 0.33389196, 0.51870899, 0.31197371],
                    [0.5772367, 0.64971211, 0.60134796, 0.39890817]],
                   [[0.40660581, 0.0459748, 0.21979608, 0.84190526],
                    [0.48786525, 0.00796778, 0.5264239, 0.49553506],
                    [0.53734049, 0.39254045, 0.24551347, 0.25985477]]], ms.float32)
    target = ms.Tensor([[[0.06027456, 0.95526571, 0.13569855, 0.41254805],
                         [0.67258379, 0.52711375, 0.76919524, 0.93621365],
                         [0.00805045, 0.06860022, 0.20532845, 0.8648434]],
                        [[0.11407711, 0.95934537, 0.43661783, 0.07770729],
                         [0.98801562, 0.78194418, 0.24647726, 0.53119156],
                         [0.30953156, 0.98123594, 0.17142974, 0.85142899]]], ms.float32)
    out0, out1, out2 = net(x, target)
    expect_out0 = [[[7.27360770e-02, 2.41565347e-01, 2.80120403e-01, 1.19376741e-01],
                    [4.05178405e-03, 1.86673272e-02, 3.13716829e-02, 1.87119961e-01],
                    [1.59593135e-01, 1.65555924e-01, 7.84157291e-02, 1.08547837e-01]],
                   [[4.27865162e-02, 3.31685275e-01, 2.35058349e-02, 2.57098973e-01],
                    [1.25075161e-01, 2.61988193e-01, 3.91850583e-02, 6.35694480e-04],
                    [2.59484462e-02, 1.69347733e-01, 2.74419948e-03, 1.70787096e-01]]]
    expect_out1 = [0.12158]
    expect_out2 = [2.91791]
    assert np.allclose(out0.asnumpy(), expect_out0)
    assert np.allclose(out1.asnumpy(), expect_out1)
    assert np.allclose(out2.asnumpy(), expect_out2)
