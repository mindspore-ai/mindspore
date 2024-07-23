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
from mindspore.ops import operations as P


class NetParallelConcat(nn.Cell):

    def __init__(self):
        super(NetParallelConcat, self).__init__()
        self.parallelconcat = P.ParallelConcat()

    def construct(self, x):
        return self.parallelconcat(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parallelconcat_1d():
    """
    Feature: ParallelConcat TEST.
    Description: 1d test case for ParallelConcat
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    x_np = (np.array([[3]])).astype(np.int8)
    y_np = (np.array([[5]])).astype(np.int8)
    z_np = np.concatenate([x_np, y_np], axis=0)

    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    net = NetParallelConcat()
    z_ms = net([x_ms, y_ms])

    assert np.allclose(z_np, z_ms.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parallelconcat_2d():
    """
    Feature: ParallelConcat TEST.
    Description: 2d test case for ParallelConcat
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    x_np = (np.array([[-1, -5, -3, -14, 64]])).astype(np.int8)
    y_np = (np.array([[5, 0, 7, 11, 66]])).astype(np.int8)
    z_np = np.concatenate([x_np, y_np], axis=0)

    x_ms = Tensor(x_np)
    y_ms = Tensor(y_np)
    net = NetParallelConcat()
    z_ms = net([x_ms, y_ms])

    assert np.allclose(z_np, z_ms.asnumpy())
