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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


class NetOnesLike(nn.Cell):
    def __init__(self):
        super(NetOnesLike, self).__init__()
        self.ones_like = P.OnesLike()

    def construct(self, x):
        return self.ones_like(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("nptype", [np.bool_, np.int8, np.int16, np.int32, np.int64,
                                    np.uint8, np.uint16, np.uint32, np.uint64,
                                    np.float16, np.float32, np.float64,
                                    np.complex64, np.complex128])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_ones_like(nptype, mode):
    """
    Feature: ALL To ALL
    Description: test cases for OnesLike
    Expectation: the result match to numpy
    """
    x0_np = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(nptype)
    x1_np = np.random.uniform(-2, 2, 1).astype(nptype)

    x0 = Tensor(x0_np)
    x1 = Tensor(x1_np)

    context.set_context(mode=mode, device_target="GPU")
    ones_like = NetOnesLike()
    output0 = ones_like(x0)
    expect0 = np.ones_like(x0_np)
    assert np.allclose(output0.asnumpy(), expect0, 1.0e-3, 1.0e-3)

    output1 = ones_like(x1)
    expect1 = np.ones_like(x1_np)
    assert np.allclose(output1.asnumpy(), expect1, 1.0e-3, 1.0e-3)
