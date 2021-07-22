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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import composite as C

def maskedselect():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]]).astype(np.bool)
    net = P.MaskedSelect()
    return net(Tensor(x), Tensor(mask))

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_maskedselect():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    y = maskedselect()
    expect = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    assert (y.asnumpy() == expect).all()


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, x, mask, grad):
        gout = self.grad(self.network)(x, mask, grad)
        return gout

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = P.MaskedSelect()

    def construct(self, x, mask):
        return self.op(x, mask)

def masked_select_grad():
    x = np.array([1, 2, 3, 4]).astype(np.int32)
    mask = np.array([[0], [1], [0], [1]]).astype(np.bool)
    dy = np.array([i for i in range(8)]).astype(np.int32)
    grad = Grad(Net())
    return grad(Tensor(x), Tensor(mask), Tensor(dy))[0]


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_masked_select_grad():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dx = masked_select_grad()
    expect = [4, 6, 8, 10]
    assert (dx.asnumpy() == expect).all()
