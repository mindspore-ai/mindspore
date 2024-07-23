# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore
from mindspore import Tensor, nn, ops, Parameter
from mindspore import context


class ApplyMomentum(nn.Cell):
    def __init__(self):
        super(ApplyMomentum, self).__init__()
        self.apply_momentum = ops.ApplyMomentum()
        self.variable = Parameter(Tensor(np.array([[0.6, 0.4], [0.1, 0.5]])
                                         .astype(np.float32)), name="variable")
        self.accumulate = Parameter(Tensor(np.array([[0.6, 0.5], [0.2, 0.6]])
                                           .astype(np.float32)), name="accumulate")

    def construct(self, lr, grad, moment):
        out = self.apply_momentum(self.variable, self.accumulate, lr, grad, moment)
        return out


def get_output(lr, grad, moment, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    if enable_graph_kernel:
        context.set_context(graph_kernel_flags="--enable_expand_ops=ApplyMomentum")
    net = ApplyMomentum()
    output = net(lr, grad, moment)
    return output, net.variable, net.accumulate


def run_apply_momentum():
    lr = Tensor(0.1, mindspore.float32)
    moment = Tensor(0.9, mindspore.float32)
    grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
    expect = get_output(lr, grad, moment, False)
    output = get_output(lr, grad, moment, True)
    for i in range(2):
        assert np.allclose(expect[i].asnumpy(), output[i].asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_momentum_ascend():
    """
    Feature: test graph kernel ApplyMomentum
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE)
    run_apply_momentum()
