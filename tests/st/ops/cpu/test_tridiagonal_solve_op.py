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

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, context
from mindspore.ops.operations import math_ops as P


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = P.TridiagonalSolve()

    def construct(self, diagonals, rhs):
        return self.op(diagonals, rhs)


def dyn_case():
    net = Net()
    diagonals_dyn = Tensor(shape=[None, None], dtype=ms.float32)
    rhs_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(diagonals_dyn, rhs_dyn)

    diagonals = Tensor(
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0,
                                                     5.0]]).astype(np.float32))
    rhs = Tensor(np.array([[1.0], [2.0], [3.0]]).astype(np.float32))
    output = net(diagonals, rhs)

    assert output.asnumpy().shape == (3, 1)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_tridiagonal_solve_dyn():
    """
    Feature: test TridiagonalSolve ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()
