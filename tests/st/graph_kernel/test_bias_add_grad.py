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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.operations import _grad_ops as G


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add_grad = G.BiasAddGrad()

    @jit
    def construct(self, dout):
        return self.bias_add_grad(dout)


def get_output(dout, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    opt = Net()
    output = opt(Tensor(dout))
    return output


def run_bias_add_grad(shape, dtype):
    np.random.seed(0)
    dout = np.random.normal(0, 1, shape).astype(dtype)

    expect = get_output(dout, False)
    output = get_output(dout, True)

    rtol = 1.e-4
    atol = 1.e-4
    if dtype == "float16":
        rtol = 1.e-3
        atol = 1.e-3
    assert np.allclose(expect.asnumpy(), output.asnumpy(), rtol, atol, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bias_add_grad_ascend():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_bias_add_grad([2, 32, 48, 64], np.float32)
    run_bias_add_grad([2, 32, 48, 64], np.float16)
