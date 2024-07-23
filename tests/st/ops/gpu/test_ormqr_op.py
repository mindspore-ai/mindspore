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
from mindspore import nn, Tensor
import mindspore.context as context
from mindspore.ops.operations.math_ops import Ormqr


class OrmqrNet(nn.Cell):
    def __init__(self, left=True, transpose=False):
        super(OrmqrNet, self).__init__()
        self.ormqr = Ormqr(left=left, transpose=transpose)

    def construct(self, x, tau, other):
        return self.ormqr(x, tau, other)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ormqr_rank2_right_double_fp():
    """
    Feature: Ormqr operator.
    Description: test cases for Ormqr: left=False, transpose=False.
    Expectation: the result match expectation.
    """
    x_np = np.array([[-114.6, 10.9, 1.1],
                     [-0.304, 38.07, 69.38],
                     [-0.45, -0.17, 62]]).astype(np.float64)
    tau_np = np.array([15.5862, 10.6579]).astype(np.float64)
    other_np = np.array([[15.5862, 10.6579, 63.8084],
                         [0.1885, -10.0553, 4.4496]]).astype(np.float64)
    expect = np.array([[270.6946003, 553.6791758, -156.4879992],
                       [-19.1850094, 64.0901946, 1.5641681]]).astype(np.float64)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = OrmqrNet(False, False)
    output_gr = net(Tensor(x_np), Tensor(tau_np), Tensor(other_np)).asnumpy()
    assert np.allclose(expect, output_gr, rtol=1.e-5, atol=1.e-5)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(Tensor(x_np), Tensor(tau_np), Tensor(other_np)).asnumpy()
    assert np.allclose(expect, output_py, rtol=1.e-5, atol=1.e-5)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ormqr_rank3_left_double_fp():
    """
    Feature: Ormqr operator.
    Description: test cases for Ormqr: left=True, transpose=False.
    Expectation: the result match expectation.
    """
    x_np = np.array([[[1.1090, -1.4204],
                      [11.4252, -3.1697],
                      [-0.5425, -0.1447]],

                     [[7.3681, -0.0566],
                      [2.8972, 5.1619],
                      [3.3822, 0.5040]]]).astype(np.float64)
    tau_np = np.array([[15.5862, 10.6579], [0.1885, -10.0553]]).astype(np.float64)
    other_np = np.array([[[0.8128, 0.6689],
                          [0.8259, 0.0635],
                          [-8.0096, -0.1519]],

                         [[10.6672, 1.0428],
                          [6.7381, 3.4068],
                          [0.3646, 6.7011]]]).astype(np.float64)
    expect = np.array([[[3566.3712760, 140.9990169],
                        [40716.8898503, 1602.4521151],
                        [-1939.2639809, -76.1491614]],

                       [[-55.6311772, -64.4607712],
                        [-115.7401958, -118.1534389],
                        [-188.7906847, -180.4638580]]]).astype(np.float64)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = OrmqrNet()
    output_gr = net(Tensor(x_np), Tensor(tau_np), Tensor(other_np)).asnumpy()
    assert np.allclose(expect, output_gr, rtol=1.e-5, atol=1.e-5)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(Tensor(x_np), Tensor(tau_np), Tensor(other_np)).asnumpy()
    assert np.allclose(expect, output_py, rtol=1.e-5, atol=1.e-5)
