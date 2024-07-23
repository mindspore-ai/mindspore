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

import torch
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


class MvlgammaNet(nn.Cell):
    def __init__(self, nptype, p):
        super(MvlgammaNet, self).__init__()
        self.mvlgamma = P.Mvlgamma(p=p)
        self.a_np = np.array([[3, 4, 5], [4, 2, 6]]).astype(nptype)
        self.a = Tensor(self.a_np)

    @jit
    def construct(self):
        return self.mvlgamma(self.a)


def mvlgamma_torch(a, d):
    return torch.mvlgamma(torch.tensor(a), d).numpy()


def mvlgamma(nptype, p):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    mvlgamma_ = MvlgammaNet(nptype, p)
    mvlgamma_output = mvlgamma_().asnumpy()
    mvlgamma_expect = mvlgamma_torch(mvlgamma_.a_np, p).astype(nptype)
    assert np.allclose(mvlgamma_output, mvlgamma_expect)


def mvlgamma_pynative(nptype, p):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    mvlgamma_ = MvlgammaNet(nptype, p)
    mvlgamma_output = mvlgamma_().asnumpy()
    mvlgamma_expect = mvlgamma_torch(mvlgamma_.a_np, p).astype(nptype)
    assert np.allclose(mvlgamma_output, mvlgamma_expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mvlgamma_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for Mvlgamma
    Expectation: the result match to numpy
    """
    mvlgamma(np.float32, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mvlgamma_pynative_float64():
    """
    Feature: ALL To ALL
    Description: test cases for Mvlgamma
    Expectation: the result match to numpy
    """
    mvlgamma_pynative(np.float64, 3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mvlgamma_functional_api_modes(mode):
    """
    Feature: Test mvlgamma functional api.
    Description: Test mvlgamma functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor([[3, 4, 5], [4, 2, 6]], mstype.float32)
    output = F.mvlgamma(x, p=3)
    expected = np.array([[2.694925, 5.402975, 9.140645], [5.402975, 1.596312, 13.64045]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mvlgamma_tensor_api_modes(mode):
    """
    Feature: Test mvlgamma tensor api.
    Description: Test mvlgamma tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor([[3, 4, 5], [4, 2, 6]], mstype.float32)
    output = x.mvlgamma(p=3)
    expected = np.array([[2.694925, 5.402975, 9.140645], [5.402975, 1.596312, 13.64045]], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_mvlgamma_tensor_element(mode):
    """
    Feature: Test mvlgamma tensor api.
    Description: Test mvlgamma tensor api for Graph and PyNative modes when the element not greater than (p-1)/2.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor([[1, 4, 5], [4, 2, 6]], mstype.float32)
    try:
        x.mvlgamma(p=3)
    except ValueError as e:
        assert "all elements of 'x' must be greater than (p-1)/2" in str(e)

    x = Tensor(np.full((7, 4, 8), np.nan), mstype.float32)
    try:
        x.mvlgamma(p=1)
    except ValueError as e:
        assert "all elements of 'x' must be greater than (p-1)/2" in str(e)
