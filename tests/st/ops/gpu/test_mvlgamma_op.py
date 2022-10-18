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

import torch
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations.array_ops as P
from mindspore import Tensor
from mindspore.common.api import jit


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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mvlgamma_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for Mvlgamma
    Expectation: the result match to numpy
    """
    mvlgamma(np.float32, 3)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mvlgamma_pynative_float64():
    """
    Feature: ALL To ALL
    Description: test cases for Mvlgamma
    Expectation: the result match to numpy
    """
    mvlgamma_pynative(np.float64, 3)
