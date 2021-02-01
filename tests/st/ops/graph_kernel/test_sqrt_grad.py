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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sqrt_grad = G.SqrtGrad()

    def construct(self, x, dout):
        return self.sqrt_grad(x, dout)


def get_output(x, dout, enable_graph_kernel=False):
    if enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    net = Net()
    output = net(x, dout)
    return output


def test_sqrt_grad(shape_x, shape_dout, dtype):
    x = Tensor(np.random.normal(0, 1, shape_x).astype(dtype))
    dout = Tensor(np.random.normal(0, 1, shape_dout).astype(dtype))

    expect = get_output(x, dout, False)
    output = get_output(x, dout, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    rtol = 0.0001
    atol = 0.0001
    if dtype == np.float16:
        rtol = 0.001
        atol = 0.001

    assert np.allclose(expect_np, output_np, rtol, atol)


def test_sqrt_grad_ascend():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    test_sqrt_grad((16, 16), (16, 16), np.float16)
    test_sqrt_grad((16, 16), (16, 16), np.float32)
