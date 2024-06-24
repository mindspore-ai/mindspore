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
from mindspore.ops.operations.math_ops import Orgqr


RTOL = 1.e-5
ATOL = 1.e-6


class OrgqrNet(nn.Cell):
    def __init__(self):
        super(OrgqrNet, self).__init__()
        self.orgqr = Orgqr()

    def construct(self, x, tau):
        return self.orgqr(x, tau)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_orgqr_rank2_double_fp():
    """
    Feature: Orgqr operator.
    Description: test cases for Orgqr operator.
    Expectation: the result match expectation.
    """
    x_np = np.array([[15.5862, 10.6579],
                     [0.1885, -10.0553],
                     [4.4496, 0.7312]]).astype(np.float64)
    tau_np = np.array([15.5862, 10.6579]).astype(np.float64)
    expect = np.array([[-14.5862000, 568.8417212],
                       [-2.9379987, 97.5687645],
                       [-69.3523555, 2523.3250663]]).astype(np.float64)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = OrgqrNet()
    output_gr = net(Tensor(x_np), Tensor(tau_np)).asnumpy()
    assert np.allclose(expect, output_gr, rtol=RTOL, atol=ATOL)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(Tensor(x_np), Tensor(tau_np)).asnumpy()
    assert np.allclose(expect, output_py, rtol=RTOL, atol=ATOL)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_orgqr_rank2_complex64_fp():
    """
    Feature: Orgqr operator.
    Description: test cases for Orgqr operator.
    Expectation: the result match expectation.
    """
    x_np = np.array([[15.5862 - 4.1226j, 10.6579 - 10.0797j],
                     [0.1885 + 10.0417j, -10.0553 - 14.9510j],
                     [4.4496 + 6.8528j, 0.7312 - 0.3177j]]).astype(np.complex64)
    tau_np = np.array([15.5862 + 5.8621j, 10.6579 + 4.9628j]).astype(np.complex64)
    expect = np.array([[-1.4586200e+01 - 5.8621001e+00j, 2.4139565e+03 - 1.9239688e+03j],
                       [5.5927452e+01 - 1.5761696e+02j, 1.9765291e+04 + 2.3872598e+04j],
                       [-2.9180559e+01 - 1.3289311e+02j, 2.3916344e+04 + 7.9812275e+03j]]).astype(np.complex64)

    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = OrgqrNet()
    output_gr = net(Tensor(x_np), Tensor(tau_np)).asnumpy()
    assert np.allclose(expect, output_gr, rtol=RTOL, atol=ATOL)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output_py = net(Tensor(x_np), Tensor(tau_np)).asnumpy()
    assert np.allclose(expect, output_py, rtol=RTOL, atol=ATOL)
