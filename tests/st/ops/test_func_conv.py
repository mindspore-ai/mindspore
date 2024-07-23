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
from mindspore import Tensor
from mindspore import ops


class Net1d(nn.Cell):
    def __init__(self, pad_mode="valid", padding=0, stride=1):
        super().__init__()
        self.pad_mode = pad_mode
        self.padding = padding
        self.stride = stride

    def construct(self, x, weight, dilation=1, groups=1, bias=None):
        return ops.conv1d(x, weight, bias, self.stride, self.pad_mode, self.padding, dilation, groups)


class Net2d(nn.Cell):
    def __init__(self, pad_mode="valid", padding=0, stride=1):
        super().__init__()
        self.pad_mode = pad_mode
        self.padding = padding
        self.stride = stride

    def construct(self, x, weight, dilation=1, groups=1, bias=None):
        return ops.conv2d(x, weight, bias, self.stride, self.pad_mode, self.padding, dilation, groups)


class Net3d(nn.Cell):
    def __init__(self, pad_mode="valid", padding=0, stride=1):
        super().__init__()
        self.pad_mode = pad_mode
        self.padding = padding
        self.stride = stride

    def construct(self, x, weight, dilation=1, groups=1, bias=None):
        return ops.conv3d(x, weight, bias, self.stride, self.pad_mode, self.padding, dilation, groups)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conv1d(mode):
    """
    Feature: ops.conv1d with padding = (1, )
    Description: Verify the result of conv1d
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.arange(32).reshape((4, 2, 4)), ms.float32)
    weight = Tensor(np.arange(8).reshape((2, 2, 2)), ms.float32)
    bias = Tensor([-0.12345, 2.7683], ms.float32)
    net = Net1d(pad_mode='pad', padding=(1,))
    output = net(x, weight, bias=bias, groups=1)
    expected = np.array([[[11.8765, 23.8766, 29.8765, 35.8765, 13.8765],
                          [30.7683, 66.7683, 88.7683, 110.7683, 56.7683]],
                         [[43.8765, 71.8765, 77.8765, 83.8765, 29.8766],
                          [126.7683, 242.7683, 264.7683, 286.7683, 136.7683]],
                         [[75.8765, 119.8765, 125.8765, 131.8766, 45.8765],
                          [222.7683, 418.7683, 440.7683, 462.7683, 216.7683]],
                         [[107.8765, 167.8766, 173.8766, 179.8766, 61.8765],
                          [318.7683, 594.7683, 616.7683, 638.7683, 296.7683]]])
    assert np.allclose(output.asnumpy(), expected, atol=1e-5, rtol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('pad_mode', ['valid', 'same', 'pad'])
def test_ops_conv2d(mode, pad_mode):
    """
    Feature: ops.conv2d
    Description: Verify the result of conv2d
    Expectation: success
    Note: There is a precision problem on Ascend, #I6PT9L
    """
    ms.set_context(mode=mode)
    x = Tensor([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                 [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]],
                [[[18.0, 19.0, 20.0], [21.0, 22.0, 23.0], [24.0, 25.0, 26.0]],
                 [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0], [33.0, 34.0, 35.0]]],
                [[[36.0, 37.0, 38.0], [39.0, 40.0, 41.0], [42.0, 43.0, 44.0]],
                 [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0], [51.0, 52.0, 53.0]]]], ms.float32)
    bias = Tensor([0.7297250824055579, 0.6472988621466479], ms.float32)
    if pad_mode == 'valid':
        net = Net2d(pad_mode)
        weight = Tensor([[[[-1.090221803810641]], [[-0.044567894776783905]]],
                         [[[0.04005113957734308]], [[0.22892450020231897]]]], ms.float32)
        output = net(x, weight, bias=bias)
        expected = np.array([[[[0.3286140263080597, -0.8061756491661072, -1.940965175628662],
                               [-3.0757548809051514, -4.210544586181641, -5.345334529876709],
                               [-6.480123996734619, -7.6149139404296875, -8.749703407287598]],
                              [[2.7076191902160645, 2.976594924926758, 3.245570659637451],
                               [3.5145461559295654, 3.783521890640259, 4.052497386932373],
                               [4.321473121643066, 4.59044885635376, 4.859424591064453]]],
                             [[[-20.097599029541016, -21.232391357421875, -22.36717987060547],
                               [-23.501968383789062, -24.63675880432129, -25.771547317504883],
                               [-26.90633773803711, -28.041128158569336, -29.17591667175293]],
                              [[7.54918098449707, 7.8181562423706055, 8.087132453918457],
                               [8.356107711791992, 8.625083923339844, 8.894059181213379],
                               [9.163034439086914, 9.432010650634766, 9.7009859085083]]],
                             [[[-40.52381134033203, -41.658599853515625, -42.79339599609375],
                               [-43.928184509277344, -45.06297302246094, -46.19776153564453],
                               [-47.33255386352539, -48.467342376708984, -49.60213088989258]],
                              [[12.390742301940918, 12.659717559814453, 12.928693771362305],
                               [13.19766902923584, 13.466644287109375, 13.735620498657227],
                               [14.004595756530762, 14.273571968078613, 14.542547225952148]]]])
        assert np.allclose(output.asnumpy(), expected, atol=1e-5, rtol=1e-5)
    elif pad_mode == 'pad':
        net = Net2d(pad_mode=pad_mode, padding=(1, 1), stride=2)
        weight = Tensor(np.arange(16).reshape((2, 2, 2, 2)), ms.float32)
        output = net(x, weight, bias=bias)
        expected = np.array([[[[63.7297, 145.7297], [186.7297, 380.7297]],
                              [[135.6473, 337.6473], [474.6473, 1052.6473]]],
                             [[[243.7297, 469.7297], [474.7297, 884.7297]],
                              [[603.6473, 1237.6472], [1338.6472, 2708.6472]]],
                             [[[423.7297, 793.7297], [762.7297, 1388.7297]],
                              [[1071.6473, 2137.6475], [2202.6475, 4364.6475]]]])
        assert np.allclose(output.asnumpy(), expected, atol=1e-5, rtol=1e-5)
    else:
        net = Net2d(pad_mode=pad_mode)
        weight = Tensor(np.arange(16).reshape((2, 2, 2, 2)), ms.float32)
        output = net(x, weight, bias=bias)
        expected = np.array([[[[268.7297, 296.7297, 138.7297],
                               [352.7297, 380.7297, 174.7297],
                               [147.7297, 157.7297, 68.7297]],
                              [[684.6473, 776.6473, 394.6473],
                               [960.6473, 1052.6473, 526.6473],
                               [499.6473, 541.6473, 268.6473]]],
                             [[[772.7297, 800.7297, 354.7297],
                               [856.7297, 884.7297, 390.7297],
                               [327.7297, 337.7297, 140.7297]],
                              [[2340.6472, 2432.6472, 1186.6472],
                               [2616.6472, 2708.6472, 1318.6472],
                               [1255.6472, 1297.6472, 628.6473]]],
                             [[[1276.7297, 1304.7297, 570.7297],
                               [1360.7297, 1388.7297, 606.7297],
                               [507.7297, 517.7297, 212.7297]],
                              [[3996.6475, 4088.6475, 1978.6473],
                               [4272.6475, 4364.6475, 2110.6475],
                               [2011.6473, 2053.6475, 988.6473]]]])
        assert np.allclose(output.asnumpy(), expected, atol=1e-5, rtol=1e-5)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_conv3d(mode):
    """
    Feature: ops.conv3d
    Description: Verify the result of conv3d
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.arange(3 * 2 * 4 * 4 * 4).reshape((3, 2, 4, 4, 4)), ms.float32)
    weight = Tensor(np.arange(32).reshape((2, 2, 2, 2, 2)), ms.float32)
    bias = Tensor(np.array([-0.12345, 2.7683]), ms.float32)
    net = Net3d(pad_mode='valid')
    output = net(x, weight, bias=bias)
    expect_output = np.array([[[[[7439.8765, 7559.8765, 7679.8765],
                                 [7919.8765, 8039.8765, 8159.8765],
                                 [8399.8770, 8519.8770, 8639.8770]],
                                [[9359.8770, 9479.8770, 9599.8770],
                                 [9839.8770, 9959.8770, 10079.8770],
                                 [10319.8770, 10439.8770, 10559.8770]],
                                [[11279.8770, 11399.8770, 11519.8770],
                                 [11759.8770, 11879.8770, 11999.8770],
                                 [12239.8770, 12359.8770, 12479.8770]]],
                               [[[18322.7695, 18698.7695, 19074.7695],
                                 [19826.7695, 20202.7695, 20578.7695],
                                 [21330.7695, 21706.7695, 22082.7695]],
                                [[24338.7695, 24714.7695, 25090.7695],
                                 [25842.7695, 26218.7695, 26594.7695],
                                 [27346.7695, 27722.7695, 28098.7695]],
                                [[30354.7695, 30730.7695, 31106.7695],
                                 [31858.7695, 32234.7695, 32610.7695],
                                 [33362.7695, 33738.7695, 34114.7695]]]],
                              [[[[22799.8770, 22919.8770, 23039.8770],
                                 [23279.8770, 23399.8770, 23519.8770],
                                 [23759.8770, 23879.8770, 23999.8770]],
                                [[24719.8770, 24839.8770, 24959.8770],
                                 [25199.8770, 25319.8770, 25439.8770],
                                 [25679.8770, 25799.8770, 25919.8770]],
                                [[26639.8770, 26759.8770, 26879.8770],
                                 [27119.8770, 27239.8770, 27359.8770],
                                 [27599.8770, 27719.8770, 27839.8770]]],
                               [[[66450.7656, 66826.7656, 67202.7656],
                                 [67954.7656, 68330.7656, 68706.7656],
                                 [69458.7656, 69834.7656, 70210.7656]],
                                [[72466.7656, 72842.7656, 73218.7656],
                                 [73970.7656, 74346.7656, 74722.7656],
                                 [75474.7656, 75850.7656, 76226.7656]],
                                [[78482.7656, 78858.7656, 79234.7656],
                                 [79986.7656, 80362.7656, 80738.7656],
                                 [81490.7656, 81866.7656, 82242.7656]]]],
                              [[[[38159.8750, 38279.8750, 38399.8750],
                                 [38639.8750, 38759.8750, 38879.8750],
                                 [39119.8750, 39239.8750, 39359.8750]],
                                [[40079.8750, 40199.8750, 40319.8750],
                                 [40559.8750, 40679.8750, 40799.8750],
                                 [41039.8750, 41159.8750, 41279.8750]],
                                [[41999.8750, 42119.8750, 42239.8750],
                                 [42479.8750, 42599.8750, 42719.8750],
                                 [42959.8750, 43079.8750, 43199.8750]]],
                               [[[114578.7656, 114954.7656, 115330.7656],
                                 [116082.7656, 116458.7656, 116834.7656],
                                 [117586.7656, 117962.7656, 118338.7656]],
                                [[120594.7656, 120970.7656, 121346.7656],
                                 [122098.7656, 122474.7656, 122850.7656],
                                 [123602.7656, 123978.7656, 124354.7656]],
                                [[126610.7656, 126986.7656, 127362.7656],
                                 [128114.7656, 128490.7656, 128866.7656],
                                 [129618.7656, 129994.7656, 130370.7656]]]]])
    assert np.allclose(output.asnumpy(), expect_output, atol=1e-5, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_conv1d_with_bf16():
    """
    Feature: The weight init of conv 1d with type of bfloat16.
    Description: The weight init of conv 1d is implemented by numpy, test type of bfloat16.
    Expectation: Success.
    """
    weight_init = ms.Tensor(np.ones([2, 2, 4]), ms.bfloat16)
    net = ms.nn.Conv1d(2, 2, 4, has_bias=False, weight_init=weight_init)
    x = ms.Tensor(np.ones([1, 2, 3]), ms.bfloat16)
    output = net(x)
    expected = [[[6., 6., 4.],
                 [6., 6., 4.]]]
    cpu_cast = ops.Cast().set_device("CPU")
    output = cpu_cast(output, ms.float32)
    assert np.allclose(output.asnumpy(), expected)
