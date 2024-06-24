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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.math_ops import Polygamma

context.set_context(mode=context.GRAPH_MODE)


class PolygammaNet(nn.Cell):

    def __init__(self):
        super(PolygammaNet, self).__init__()
        self.polygamma = Polygamma()

    def construct(self, a, x):
        return self.polygamma(a, x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_polygamma_1d_a_1_int64_float16():
    """
    Feature: Polygamma
    Description: test cases for Polygamma of float16
    Expectation: the result match to torch
    """
    net = PolygammaNet()
    a = np.array(1).astype(np.int64)
    x_ms = np.array([1, 0.4273, 9, -3.12, 12246.345]).astype(np.float16)
    z_ms = net(Tensor(a), Tensor(x_ms))
    expect = np.array([1.64493407e+00, 6.47734100e+00, 1.17512015e-01, 7.35594209e+01,
                       8.16493161e-05]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect.astype(np.float16), 0.001, 0.001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_polygamma_1d_a_1_int64_float32():
    """
    Feature: Polygamma
    Description: test cases for Polygamma of float16
    Expectation: the result match to torch
    """
    net = PolygammaNet()
    a = np.array(1).astype(np.int64)
    x_ms = np.array([1, 0.5273, 9, -3.12, 13250]).astype(np.float32)
    z_ms = net(Tensor(a), Tensor(x_ms))
    expect = np.array([1.6449341e+00, 4.5092258e+00, 1.1751202e-01, 7.2555374e+01,
                       7.5474542e-05]).astype(np.float32)
    assert np.allclose(z_ms.asnumpy(), expect, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_polygamma_1d_a_1_int64_float64():
    """
    Feature: Polygamma
    Description: test cases for Polygamma of float64
    Expectation: the result match to torch
    """
    net = PolygammaNet()
    a = np.array(1).astype(np.int64)
    x_ms = np.array([1, 0.5273, 9, -3.12, 13250]).astype(np.float64)
    z_ms = net(Tensor(a), Tensor(x_ms))
    expect = np.array([1.64493407e+00, 4.50922599e+00, 1.17512015e-01, 7.25554469e+01,
                       7.54745462e-05]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect, 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_polygamma_1d_a_2_int64_float16():
    """
    Feature: Polygamma
    Description: test cases for Polygamma of float16
    Expectation: the result match to torch
    """
    net = PolygammaNet()
    a = np.array(2).astype(np.int64)
    x_ms = np.array([1, 0.4273, 9, -3.12, 12246.345]).astype(np.float16)
    z_ms = net(Tensor(a), Tensor(x_ms))
    expect = np.array([-2.40411381e+00, -2.65858621e+01, -1.37933192e-02,
                       1.18094081e+03, -6.66661082e-09]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect.astype(np.float16), 0.001, 0.001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_polygamma_1d_a_2_int64_float32():
    """
    Feature: Polygamma
    Description: test cases for Polygamma of float16
    Expectation: the result match to torch
    """
    net = PolygammaNet()
    a = np.array(2).astype(np.int64)
    x_ms = np.array([1, 0.5273, 9, -3.12, 13250]).astype(np.float64)
    z_ms = net(Tensor(a), Tensor(x_ms))
    expect = np.array([-2.40411381e+00, -1.44329154e+01, -1.37933192e-02,
                       1.15570148e+03, -5.69640712e-09]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_polygamma_1d_a_2_int64_float64():
    """
    Feature: Polygamma
    Description: test cases for Polygamma of float64
    Expectation: the result match to torch
    """
    net = PolygammaNet()
    a = np.array(2).astype(np.int64)
    x_ms = np.array([1, 0.5273, 9, -3.12, 13250]).astype(np.float64)
    z_ms = net(Tensor(a), Tensor(x_ms))
    expect = np.array([-2.40411381e+00, -1.44329154e+01, -1.37933192e-02,
                       1.15570148e+03, -5.69640712e-09]).astype(np.float64)
    assert np.allclose(z_ms.asnumpy(), expect, 0.00001, 0.00001)
