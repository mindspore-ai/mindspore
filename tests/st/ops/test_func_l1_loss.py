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
        output0 = ops.l1_loss(x, target, reduction="none")
        output1 = ops.l1_loss(x, target, reduction="mean")
        output2 = ops.l1_loss(x, target, reduction="sum")
        return output0, output1, output2


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_l1_loss(mode):
    """
    Feature: Test l1_loss
    Description: Test the functionality of l1_loss
    Expectation: Success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor([[[1.17273476, -0.05052809, 0.61813106, 0.16455488, -1.35581311],
                    [1.32487223, 0.13208311, -1.31230669, -0.50771298, 1.32278446],
                    [-0.04625993, 1.18794348, -1.21238798, 0.01314028, -1.20131357]],
                   [[-1.4510571, -1.03311918, -1.00915919, 0.6134792, 0.56710962],
                    [-1.39683892, -0.0932166, -1.06056463, 0.20178101, 0.47950521],
                    [-1.39548584, -1.70302071, -0.48198836, -0.77789908, 0.87970894]]], ms.float32)
    target = ms.Tensor([[[-1.30292448, -0.35515205, 1.48585374, 0.22724189, 0.60810377],
                         [-1.14444725, 1.90415392, 0.45537515, -1.20027348, 1.81567979],
                         [0.30801377, -0.79452551, 1.80005659, 0.98829231, 2.07602126]],
                        [[0.05371826, 0.20575326, 1.3496286, 1.55930587, -0.50407597],
                         [-1.97812696, -1.38987021, -1.95899861, -1.05986999, 0.02349943],
                         [0.25305345, 0.42477621, 1.74664105, -0.50482991, -0.24119833]]], ms.float32)
    out0, out1, out2 = net(x, target)
    expect_out0 = [[[2.47565937e+00, 3.04623961e-01, 8.67722750e-01, 6.26870096e-02, 1.96391690e+00],
                    [2.46931934e+00, 1.77207088e+00, 1.76768184e+00, 6.92560554e-01, 4.92895365e-01],
                    [3.54273707e-01, 1.98246896e+00, 3.01244450e+00, 9.75152075e-01, 3.27733469e+00]],
                   [[1.50477529e+00, 1.23887253e+00, 2.35878778e+00, 9.45826709e-01, 1.07118559e+00],
                    [5.81288099e-01, 1.29665351e+00, 8.98433924e-01, 1.26165104e+00, 4.56005782e-01],
                    [1.64853930e+00, 2.12779689e+00, 2.22862935e+00, 2.73069203e-01, 1.12090731e+00]]]
    expect_out1 = [1.3827745]
    expect_out2 = [41.483234]
    assert np.allclose(out0.asnumpy(), expect_out0)
    assert np.allclose(out1.asnumpy(), expect_out1)
    assert np.allclose(out2.asnumpy(), expect_out2)
