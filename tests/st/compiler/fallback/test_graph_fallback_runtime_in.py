# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
import torch
import numpy as np
from mindspore import Tensor, context
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_in_asnumpy():
    """
    Feature: Support in.
    Description: Support in in fallback runtime.
    Expectation: No exception.
    """
    class Net(nn.Cell):
        def construct(self, y):
            input_x = Tensor([1, 2], dtype=mstype.int32).asnumpy()
            return y in input_x

    net = Net()
    res = net(2)
    assert res


class GradNet(nn.Cell):
    def __init__(self, net, grad_position=0):
        super().__init__()
        # pylint: disable=E1102
        self.grad_net = ops.grad(net, grad_position=grad_position)

    def construct(self, *x):
        return self.grad_net(*x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_fallback_parse_node_conver_in_not_in():
    """
    Feature: Support in.
    Description: Support 'in' in fallback runtime.
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self, x):
            if (None, 1) not in ((None, 1), 1, 2, 3):
                out = x
            elif (None, 2) in ((None, 1), 1, 2, 3):
                out = x+x
            else:
                out = x+x+x
            return out

    class TouchNet(torch.nn.Module):
        def forward(self, x):
            if (None, 1) not in ((None, 1), 1, 2, 3):
                out = x
            elif (None, 2) in ((None, 1), 1, 2, 3):
                out = x+x
            else:
                out = x+x+x
            return out

    x = np.random.randn(2, 2).astype(np.float32)
    tc_x = torch.tensor(x, requires_grad=True)

    ms_out = InnerClass()(Tensor(x))
    torch_out = TouchNet()(tc_x)
    assert np.allclose(torch_out.detach().numpy(), ms_out.asnumpy(), 1e-5, 1e-5)

    torch_out.backward(torch.ones_like(torch_out))
    tc_grad = tc_x.grad
    ms_grad = GradNet(InnerClass())(Tensor(x))
    assert np.allclose(tc_grad.numpy(), ms_grad.asnumpy(), 1e-5, 1e-5)
