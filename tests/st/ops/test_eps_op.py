# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
from mindspore.ops import operations as P


class NetEps(nn.Cell):
    def __init__(self):
        super(NetEps, self).__init__()
        self.eps = P.Eps()

    def construct(self, x):
        return self.eps(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("nptype", [np.float32])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_eps(nptype, mode):
    """
    Feature: ALL To ALL
    Description: test cases for Eps
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    x1_np = np.random.uniform(-2, 2, 1).astype(nptype)
    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)
    context.set_context(mode=mode)
    eps = NetEps()
    output0 = eps(x0)
    out_exp = np.full(x0_np.shape, np.finfo(np.float32).eps, dtype=np.float32)
    assert np.allclose(output0.asnumpy(), out_exp, 1.0e-7, 1.0e-7)
    output1 = eps(x1)
    out_exp1 = np.full(x1_np.shape, np.finfo(np.float32).eps, dtype=np.float32)
    assert np.allclose(output1.asnumpy(), out_exp1, 1.0e-7, 1.0e-7)
