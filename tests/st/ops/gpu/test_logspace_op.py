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

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations.array_ops as opa


class LogSpaceNet(Cell):
    def __init__(self, steps=10, base=10, dtype=mstype.float32):
        super(LogSpaceNet, self).__init__()
        self.ls_op = opa.LogSpace(steps, base, dtype)

    def construct(self, start, stop):
        output = self.ls_op(start, stop)
        return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_log_space():
    """
    Feature: Create a one-dimensional tensor of size steps whose values are spaced from base**start to base**end,
    Description: test cases for logspace
    Expectation: the result match to numpy
    """
    start_np = -5
    stop_np = 20
    num_np = 20
    base_np = 2
    result_np = np.logspace(start_np, stop_np, num_np, base=base_np)
    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    net_g = LogSpaceNet(num_np, base_np)
    result_g = net_g(start, stop).asnumpy()
    assert np.allclose(result_g, result_np, 1e-5, 1e-5)
