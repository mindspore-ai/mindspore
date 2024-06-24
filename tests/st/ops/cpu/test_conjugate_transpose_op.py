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

import pytest
import numpy as np

import mindspore.ops.operations.array_ops as P
from mindspore.common import dtype as mstype
from mindspore import Tensor, nn, context


class Net(nn.Cell):

    def __init__(self):
        super(Net, self).__init__()
        self.op = P.ConjugateTranspose()

    def construct(self, x, perm):
        return self.op(x, perm)


def dyn_case():
    net = Net()

    x_dyn = Tensor(shape=[None, 4, None, 7], dtype=mstype.float32)
    perm = (2, 1, 0, 3)

    net.set_inputs(x_dyn, perm)

    x = Tensor(np.random.random((8, 4, 5, 7)).astype(np.float32))
    out = net(x, perm)

    expect_shape = (5, 4, 8, 7)
    assert out.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_conjugate_transpose_dyn():
    """
    Feature: test ConjugateTranspose in PyNative and Graph modes.
    Description: test dynamic shape case.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()
