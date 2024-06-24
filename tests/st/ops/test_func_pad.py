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
    def __init__(self, mode):
        super(Net, self).__init__()
        self.mode = mode

    def construct(self, x, padding, value=None):
        output = ops.pad(x, padding, self.mode, value)
        return output


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('pad_mode', ["constant", "reflect", "replicate"])
@pytest.mark.parametrize('padding', [[1, 2, 2, 1], (1, 2, 2, 1), ms.Tensor([1, 2, 2, 1])])
def test_pad_normal(mode, pad_mode, padding):
    """
    Feature: pad
    Description: Verify the result of pad
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net(pad_mode)
    x = ms.Tensor(np.arange(1 * 2 * 3 * 4).reshape((1, 2, 3, 4)), dtype=ms.float64)

    if pad_mode == "constant":
        output = net(x, padding, 6)
        expect_output = np.array([[[[6., 6., 6., 6., 6., 6., 6.],
                                    [6., 6., 6., 6., 6., 6., 6.],
                                    [6., 0., 1., 2., 3., 6., 6.],
                                    [6., 4., 5., 6., 7., 6., 6.],
                                    [6., 8., 9., 10., 11., 6., 6.],
                                    [6., 6., 6., 6., 6., 6., 6.]],

                                   [[6., 6., 6., 6., 6., 6., 6.],
                                    [6., 6., 6., 6., 6., 6., 6.],
                                    [6., 12., 13., 14., 15., 6., 6.],
                                    [6., 16., 17., 18., 19., 6., 6.],
                                    [6., 20., 21., 22., 23., 6., 6.],
                                    [6., 6., 6., 6., 6., 6., 6.]]]])
    elif pad_mode == "reflect":
        output = net(x, padding)
        expect_output = np.array([[[[9., 8., 9., 10., 11., 10., 9.],
                                    [5., 4., 5., 6., 7., 6., 5.],
                                    [1., 0., 1., 2., 3., 2., 1.],
                                    [5., 4., 5., 6., 7., 6., 5.],
                                    [9., 8., 9., 10., 11., 10., 9.],
                                    [5., 4., 5., 6., 7., 6., 5.]],

                                   [[21., 20., 21., 22., 23., 22., 21.],
                                    [17., 16., 17., 18., 19., 18., 17.],
                                    [13., 12., 13., 14., 15., 14., 13.],
                                    [17., 16., 17., 18., 19., 18., 17.],
                                    [21., 20., 21., 22., 23., 22., 21.],
                                    [17., 16., 17., 18., 19., 18., 17.]]]])
    else:
        output = net(x, padding)
        expect_output = np.array([[[[0., 0., 1., 2., 3., 3., 3.],
                                    [0., 0., 1., 2., 3., 3., 3.],
                                    [0., 0., 1., 2., 3., 3., 3.],
                                    [4., 4., 5., 6., 7., 7., 7.],
                                    [8., 8., 9., 10., 11., 11., 11.],
                                    [8., 8., 9., 10., 11., 11., 11.]],

                                   [[12., 12., 13., 14., 15., 15., 15.],
                                    [12., 12., 13., 14., 15., 15., 15.],
                                    [12., 12., 13., 14., 15., 15., 15.],
                                    [16., 16., 17., 18., 19., 19., 19.],
                                    [20., 20., 21., 22., 23., 23., 23.],
                                    [20., 20., 21., 22., 23., 23., 23.]]]])
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('pad_mode', ["constant", "reflect", "replicate"])
@pytest.mark.parametrize('padding', [[-1, 2, 2, 1]])
def test_pad_negative(mode, pad_mode, padding):
    """
    Feature: pad
    Description: Verify the result of pad when padding is negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net(pad_mode)
    x = ms.Tensor(np.arange(1 * 2 * 3 * 4).reshape((1, 2, 3, 4)), dtype=ms.float64)

    if pad_mode == "constant":
        output = net(x, padding, 6)
        expect_output = np.array([[[[6., 6., 6., 6., 6.],
                                    [6., 6., 6., 6., 6.],
                                    [1., 2., 3., 6., 6.],
                                    [5., 6., 7., 6., 6.],
                                    [9., 10., 11., 6., 6.],
                                    [6., 6., 6., 6., 6.]],

                                   [[6., 6., 6., 6., 6.],
                                    [6., 6., 6., 6., 6.],
                                    [13., 14., 15., 6., 6.],
                                    [17., 18., 19., 6., 6.],
                                    [21., 22., 23., 6., 6.],
                                    [6., 6., 6., 6., 6.]]]])
    elif pad_mode == "reflect":
        output = net(x, padding)
        expect_output = np.array([[[[9., 10., 11., 10., 9.],
                                    [5., 6., 7., 6., 5.],
                                    [1., 2., 3., 2., 1.],
                                    [5., 6., 7., 6., 5.],
                                    [9., 10., 11., 10., 9.],
                                    [5., 6., 7., 6., 5.]],

                                   [[21., 22., 23., 22., 21.],
                                    [17., 18., 19., 18., 17.],
                                    [13., 14., 15., 14., 13.],
                                    [17., 18., 19., 18., 17.],
                                    [21., 22., 23., 22., 21.],
                                    [17., 18., 19., 18., 17.]]]])

    else:
        output = net(x, padding)
        expect_output = np.array([[[[1., 2., 3., 3., 3.],
                                    [1., 2., 3., 3., 3.],
                                    [1., 2., 3., 3., 3.],
                                    [5., 6., 7., 7., 7.],
                                    [9., 10., 11., 11., 11.],
                                    [9., 10., 11., 11., 11.]],

                                   [[13., 14., 15., 15., 15.],
                                    [13., 14., 15., 15., 15.],
                                    [13., 14., 15., 15., 15.],
                                    [17., 18., 19., 19., 19.],
                                    [21., 22., 23., 23., 23.],
                                    [21., 22., 23., 23., 23.]]]])
    assert np.allclose(output.asnumpy(), expect_output)
