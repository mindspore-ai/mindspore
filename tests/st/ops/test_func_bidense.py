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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x, y, weight, bias):
        output = ms.ops.bidense(x, y, weight, bias)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bidense_two_dim(mode):
    """
    Feature: ops.bidense
    Description: Verify the result of bidense
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[-1.1283, 1.2603],
                   [0.0214, 0.7801],
                   [-1.2086, 1.2849]], ms.float32)
    y = ms.Tensor([[-0.4631, 0.3238, 0.4201],
                   [0.6215, -1.0910, -0.5757],
                   [-0.7788, -0.0706, -0.7942]], ms.float32)
    weight = ms.Tensor([[[-0.3132, 0.9271, 1.1010],
                         [0.6555, -1.2162, -0.2987]],
                        [[1.0458, 0.5886, 0.2523],
                         [-1.3486, -0.8103, -0.2080]],
                        [[1.1685, 0.5569, -0.3987],
                         [-0.4265, -2.6295, 0.8535]],
                        [[0.6948, -1.1288, -0.6978],
                         [0.3511, 0.0609, -0.1122]]], ms.float32)
    net = Net()
    output = net(x, y, weight, None)
    expect_output = [[-2.0611, 0.5582, 0.2241, 0.8666],
                     [1.4478, 0.1263, 1.6554, 0.2128],
                     [0.6004, 2.9122, 0.5592, -0.3545]]
    rtol = 1e-03
    atol = 1e-03
    assert np.allclose(output.asnumpy(), expect_output, rtol, atol)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bidense_three_dim(mode):
    """
    Feature: ops.bidense
    Description: Verify the result of bidense
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[[0.4592, 0.7692],
                    [-1.5433, 0.8188],
                    [-0.3386, -0.6108]],
                   [[0.1843, 0.9144],
                    [1.1210, -0.0436],
                    [0.1125, -1.3160]]], ms.float32)

    y = ms.Tensor([[[-0.3663, -2.0205, 0.0294],
                    [0.4309, 0.7456, -0.6703],
                    [2.0860, 0.2569, 1.3150]],
                   [[-1.2014, 1.0051, -0.3922],
                    [-1.4815, -1.3695, 0.6552],
                    [-0.4056, -0.4768, 1.9908]]], ms.float32)

    weight = ms.Tensor([[[-0.9672, 1.6880, -0.3678],
                         [-0.0100, -0.0742, -0.2186]],
                        [[-0.7013, -0.2034, 0.1811],
                         [-1.9481, 1.7017, -0.0622]],
                        [[1.3461, 0.6559, 0.7194],
                         [-1.0988, -0.5309, -0.5339]],
                        [[-0.0486, 0.0237, -1.0220],
                         [0.4082, 0.7080, -0.3781]]], ms.float32)

    bias = ms.Tensor([1.1433, -1.1392, 1.3866, 0.4529], ms.float32)

    net = Net()
    output = net(x, y, weight, bias)
    expect_output = [[[-0.1520, -2.9272, 1.6839, -0.7985],
                      [-0.4652, 0.1343, 0.0621, 0.1845],
                      [2.0432, 1.5581, 1.9707, 0.6128]],
                     [[1.7180, 2.6918, 2.0688, 0.8798],
                      [-0.1106, 0.4483, -1.4148, -0.1738],
                      [1.5353, -0.8651, 1.9302, 1.8777]]]

    rtol = 1e-03
    atol = 1e-03
    assert np.allclose(output.asnumpy(), expect_output, rtol, atol)
