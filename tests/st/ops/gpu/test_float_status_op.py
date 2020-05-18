# Copyright 2020 Huawei Technologies Co., Ltd
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

import pytest
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.status = P.FloatStatus()

    def construct(self, x):
        return self.status(x)


class Netnan(nn.Cell):
    def __init__(self):
        super(Netnan, self).__init__()
        self.isnan = P.IsNan()

    def construct(self, x):
        return self.isnan(x)


class Netinf(nn.Cell):
    def __init__(self):
        super(Netinf, self).__init__()
        self.isinf = P.IsInf()

    def construct(self, x):
        return self.isinf(x)


class Netfinite(nn.Cell):
    def __init__(self):
        super(Netfinite, self).__init__()
        self.isfinite = P.IsFinite()

    def construct(self, x):
        return self.isfinite(x)


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
x1 = np.array([[1.2, 2, np.nan, 88]]).astype(np.float32)
x2 = np.array([[np.inf, 1, 88.0, 0]]).astype(np.float32)
x3 = np.array([[1, 2], [3, 4], [5.0, 88.0]]).astype(np.float32)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_status():
    ms_status = Net();
    output1 = ms_status(Tensor(x1))
    output2 = ms_status(Tensor(x2))
    output3 = ms_status(Tensor(x3))
    expect1 = 1
    expect2 = 1
    expect3 = 0
    assert output1.asnumpy()[0] == expect1
    assert output2.asnumpy()[0] == expect2
    assert output3.asnumpy()[0] == expect3


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nan():
    ms_isnan = Netnan();
    output1 = ms_isnan(Tensor(x1))
    output2 = ms_isnan(Tensor(x2))
    output3 = ms_isnan(Tensor(x3))
    expect1 = [[False, False, True, False]]
    expect2 = [[False, False, False, False]]
    expect3 = [[False, False], [False, False], [False, False]]
    assert (output1.asnumpy() == expect1).all()
    assert (output2.asnumpy() == expect2).all()
    assert (output3.asnumpy() == expect3).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inf():
    ms_isinf = Netinf();
    output1 = ms_isinf(Tensor(x1))
    output2 = ms_isinf(Tensor(x2))
    output3 = ms_isinf(Tensor(x3))
    expect1 = [[False, False, False, False]]
    expect2 = [[True, False, False, False]]
    expect3 = [[False, False], [False, False], [False, False]]
    assert (output1.asnumpy() == expect1).all()
    assert (output2.asnumpy() == expect2).all()
    assert (output3.asnumpy() == expect3).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_finite():
    ms_isfinite = Netfinite();
    output1 = ms_isfinite(Tensor(x1))
    output2 = ms_isfinite(Tensor(x2))
    output3 = ms_isfinite(Tensor(x3))
    expect1 = [[True, True, False, True]]
    expect2 = [[False, True, True, True]]
    expect3 = [[True, True], [True, True], [True, True]]
    assert (output1.asnumpy() == expect1).all()
    assert (output2.asnumpy() == expect2).all()
    assert (output3.asnumpy() == expect3).all()
