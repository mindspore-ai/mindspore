# Copyright 2021 Huawei Technologies Co., Ltd
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
"""st for scipy.optimize."""

import numpy as np
import mindspore as ms
import mindspore.scipy as msp
from mindspore import context
from mindspore.common import Tensor

from tests.mark_utils import arg_mark


def fun(d):
    x = d[0]
    y = d[1]
    return -(2*x*y + 2*x - x**2 - 2*y**2)


def f_ieqcon(x):
    return -(x[0] - x[1] - 1.0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimize_lagrange():
    """
    Feature: ALL TO ALL
    Description: test cases for lagrange in GRAPH mode
    Expectation: the result match scipy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    object_func = fun
    x0 = Tensor([-1.0, 1.0], dtype=ms.float32)
    constrain = [f_ieqcon,]
    option = dict(rounds=10, steps=50)
    res = msp.optimize.minimize(object_func, x0=x0, method="lagrange", constraints=constrain, options=option)
    expect = np.array([2., 1.])
    assert np.allclose(expect, res.best_value, 0.01, 0.01)
