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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops.operations as P
from mindspore.ops.operations import _grad_ops as G


class Net(Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.Add()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.sqrt_grad = G.SqrtGrad()

    def construct(self, x, y, z):
        sub_res = self.sub(x, y)
        mul_res = self.mul(sub_res, x)
        sqrt_grad_res = self.sqrt_grad(mul_res, z)
        square_res = P.Square()(sqrt_grad_res)
        add_res = self.add(sqrt_grad_res, square_res)
        add1_res = self.add(add_res, add_res)
        return self.add(add1_res, add1_res)


def get_output(i0, i1, i2, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    net = Net()
    output = net(i0, i1, i2)
    return output


def test_basic():
    i0 = Tensor(np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32))
    i1 = Tensor(np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32))
    i2 = Tensor(np.random.normal(0, 1, [2, 3, 4, 3]).astype(np.float32))

    expect = get_output(i0, i1, i2, False)
    output = get_output(i0, i1, i2, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 1.e-4, 1.e-7)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_basic_gpu():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_basic()


def test_basic_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_basic()
