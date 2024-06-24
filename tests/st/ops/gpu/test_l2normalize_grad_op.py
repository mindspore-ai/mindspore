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
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops.operations import _grad_ops as G

class Net(Cell):
    def __init__(self, axis=0, epsilon=1e-12):
        super(Net, self).__init__()
        self.norm_grad = G.L2NormalizeGrad(axis=axis, epsilon=epsilon)

    def construct(self, x, out, dout):
        return self.norm_grad(x, out, dout)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_l2normalize_grad():
    """
    Feature: test l2normalize_grad op in gpu.
    Description: test the ops.
    Expectation: expect correct shape result.
    """
    axis_ = 0
    x = np.random.randint(1, 10, (2, 3, 4, 4)).astype(np.float32)
    y = x / np.sqrt(np.sum(x**2, axis=axis_, keepdims=True))
    dy = np.random.randint(1, 10, (2, 3, 4, 4)).astype(np.float32)
    expect = (dy - y * np.sum(y * dy, axis=axis_, keepdims=True)) / np.sqrt(np.sum(x**2, axis=axis_, keepdims=True))
    x = Tensor(x)
    y = Tensor(y)
    dy = Tensor(dy)
    error = np.ones(shape=[2, 3, 4, 4]) * 1.0e-5

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    norm_grad_op = Net(axis=axis_)
    output = norm_grad_op(x, y, dy)
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)
