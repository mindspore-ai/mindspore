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
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops.operations.math_ops as P
from mindspore import Tensor


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = P.MatrixLogarithm()

    def construct(self, x):
        return self.op(x)


def dyn_case():
    net = Net()

    x_dyn = Tensor(shape=[None, None], dtype=ms.complex128)
    net.set_inputs(x_dyn)

    x = Tensor([[1 + 2j, 2 + 1j], [4 + 1j, 5 + 2j]])
    y = net(x)

    assert y.asnumpy().shape == (2, 2)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_matrix_logarithm_dyn():
    """
    Feature: test MatrixLogarithm in cpu.
    Description: test the ops in dynamic case.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()
