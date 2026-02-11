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
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.ops as P
from mindspore.common import ParameterTuple
import torch
import torch.nn as nn_pt
from tests.mark_utils import arg_mark

class GradofAllInputsAndParams(nn.Cell):
    def __init__(self, net, sens=False):
        super().__init__()
        self.grad = P.GradOperation(get_all=True, get_by_list=True, sens_param=sens)
        self.net = net
        self.params = ParameterTuple(self.net.trainable_params())

    def construct(self, *x):
        out = self.grad(self.net, self.params)(*x)
        return out

@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_sit_pynative_diff_shape_with_while_in_construct():
    class WhileNetMs(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3, weight_init='ones', pad_mode='pad')

        def construct(self, x, flag):
            while flag:
                if flag > 1:
                    x = self.conv(x)
                else:
                    x = x + 1
                flag = flag - 1
            return x

    class WhileNetPt(nn_pt.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn_pt.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3),
                                     stride=1, padding=0, bias=False)
            self.weight = nn_pt.Parameter(torch.from_numpy(np.ones([1, 1, 3, 3]).astype(np.float32)))
            self.conv.register_parameter('weight', self.weight)

        def forward(self, x, flag):
            while flag:
                if flag > 1:
                    x = self.conv(x)
                else:
                    x = x + 1
                flag = flag - 1
            return x

    net = WhileNetMs()
    input_ms = Tensor(np.random.rand(1, 1, 224, 224).astype(np.float32))
    flag = 2
    out = net(input_ms, flag)
    backnet = GradofAllInputsAndParams(net)
    backout = backnet(input_ms, Tensor(flag, mstype.int32))

    comparenet = WhileNetPt()
    torch_input = torch.from_numpy(input_ms.asnumpy())
    torch_input.requires_grad = True
    torch_flag = torch.from_numpy(np.array(flag))
    torch_flag.requires_grad = False
    out_good = comparenet(torch_input, torch_flag)
    grad = torch.from_numpy(np.ones_like(out_good.detach().numpy()).astype(np.float32))
    out_good.backward(gradient=grad)
    assert np.allclose(out_good.detach().numpy(), out.asnumpy(), 0.01, 0.01)
    assert np.allclose(torch_input.grad.numpy(), backout[0][0].asnumpy(), 0.01, 0.01)
    assert np.allclose(comparenet.weight.grad.numpy(), backout[1][0].asnumpy(), 0.01, 0.01)

    flag = 3
    out = net(input_ms, flag)
    backout = backnet(input_ms, Tensor(flag, mstype.int32))
    torch_flag = torch.from_numpy(np.array(flag))
    torch_flag.requires_grad = False
    comparenet.zero_grad()
    torch_input.grad.zero_()
    out_good = comparenet(torch_input, torch_flag)
    grad = torch.from_numpy(np.ones_like(out_good.detach().numpy()).astype(np.float32))
    out_good.backward(gradient=grad)
    assert np.allclose(out_good.detach().numpy(), out.asnumpy(), 0.01, 0.01)
    assert np.allclose(torch_input.grad.numpy(), backout[0][0].asnumpy(), 0.01, 0.01)
    assert np.allclose(comparenet.weight.grad.numpy(), backout[1][0].asnumpy(), 0.01, 0.01)

    input_ms = Tensor(np.random.rand(1, 1, 112, 112).astype(np.float32))
    flag = 4
    backout = backnet(input_ms, Tensor(flag, mstype.int32))
    torch_input = torch.from_numpy(input_ms.asnumpy())
    torch_input.requires_grad = True
    torch_flag = torch.from_numpy(np.array(flag))
    torch_flag.requires_grad = False
    comparenet.zero_grad()
    out_good = comparenet(torch_input, torch_flag)
    grad = torch.from_numpy(np.ones_like(out_good.detach().numpy()).astype(np.float32))
    out_good.backward(gradient=grad)
    assert np.allclose(torch_input.grad.numpy(), backout[0][0].asnumpy(), 0.01, 0.01)
    assert np.allclose(comparenet.weight.grad.numpy(), backout[1][0].asnumpy(), 0.01, 0.01)
