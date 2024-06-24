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
import torch
import pytest
import mindspore as ms
from mindspore import nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.ops.operations.linalg_ops import Geqrf


RTOL = 1.e-5
ATOL = 1.e-6


class GeqrfNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.geqrf = Geqrf()

    def construct(self, x):
        return self.geqrf(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_geqrf_rank2_double_fp():
    """
    Feature: Geqrf operator.
    Description: test cases for Geqrf operator.
    Expectation: the result match expectation.
    """
    x_np = np.array([[15.5862, 10.6579],
                     [0.1885, -10.0553],
                     [4.4496, 0.7312]]).astype(np.float64)
    expect_y, expect_tau = torch.geqrf(torch.tensor(x_np))
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = GeqrfNet()
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_geqrf_rank3_float_fp():
    """
    Feature: Geqrf operator.
    Description: test cases for Geqrf operator.
    Expectation: the result match expectation.
    """
    x_np = np.array([[[1.7705, 0.5642],
                      [0.8125, 0.4068],
                      [0.1714, -0.1203]],

                     [[-1.5411, 1.5866],
                      [-0.3722, -0.9087],
                      [-0.0862, -0.7602]]]).astype(np.float32)
    expect_y = np.array([[[-1.9555572, -0.6692831],
                          [0.2180589, -0.2243657],
                          [0.0460004, -0.4888012]],

                         [[1.5877508, -1.2856942],
                          [0.1189574, 1.5059648],
                          [0.0275500, 0.3045090]]]).astype(np.float32)
    expect_tau = np.array([[1.9053684, 1.6143007], [1.9706184, 1.8302855]]).astype(np.float32)
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = GeqrfNet()
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_geqrf_acl():
    """
    Feature: Geqrf operator.
    Description: test cases for Geqrf operator on ACL.
    Expectation: the result match expectation.
    """
    x_np = np.array([[15.5862, 10.6579],
                     [0.1885, -10.0553],
                     [4.4496, 0.7312]]).astype(np.float64)
    expect_y, expect_tau = torch.geqrf(torch.tensor(x_np))
    context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend')
    net = GeqrfNet()
    x_dyn = Tensor(shape=[None for _ in x_np.shape], dtype=ms.float64)
    net.set_inputs(x_dyn)
    y, tau = net(Tensor(x_np))
    assert np.allclose(expect_y, y.asnumpy(), rtol=RTOL, atol=ATOL)
    assert np.allclose(expect_tau, tau.asnumpy(), rtol=RTOL, atol=ATOL)