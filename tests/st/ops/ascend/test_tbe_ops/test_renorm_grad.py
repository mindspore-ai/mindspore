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
import numpy as np
import torch
import mindspore
from mindspore.ops.composite import GradOperation
from mindspore import context, nn, Tensor
from mindspore.ops.operations.math_ops import Renorm

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class RenormNet(nn.Cell):
    def __init__(self, p, dim, max_norm):
        super(RenormNet, self).__init__()
        self.p = p
        self.dim = dim
        self.max_norm = max_norm
        self.renorm = Renorm(p=self.p, dim=self.dim, maxnorm=self.max_norm)

    def construct(self, x):
        y = self.renorm(x)
        return y


class RenormGrad(nn.Cell):
    def __init__(self, network):
        super(RenormGrad, self).__init__()
        self.grad = GradOperation(get_all=True, get_by_list=False, sens_param=True)
        self.network = network

    def construct(self, x, grad):
        return self.grad(self.network)(x, grad)


class TorchNet(torch.nn.Module):
    def __init__(self, p, dim, maxnorm):
        super(TorchNet, self).__init__()
        self.p = p
        self.dim = dim
        self.maxnorm = maxnorm

    def forward(self, x):
        y = torch.renorm(x, self.p, self.dim, self.maxnorm)
        return y


a = np.random.random([2, 3, 4, 5])
tensor = Tensor(a, mindspore.float32)
out_grad = np.random.random(a.shape).astype(np.float32)


def test_grad():
    """
    Feature: test renorm grad
    Description: test renorm grad with input tensor's type float32
    Expectation: none.
    """
    p = 3
    dim = -2
    max_norm = 5.
    ms_net = RenormNet(p, dim, max_norm)
    grad_net = RenormGrad(ms_net)
    grad_net.set_train()
    grad_out = grad_net(tensor, Tensor(out_grad, dtype=mindspore.float32))

    torch_x = torch.tensor(a, dtype=torch.float32)
    torch_x.requires_grad = True
    torch_net = TorchNet(p, dim, max_norm)
    torch_out = torch_net(torch_x)
    torch_out.backward(torch.from_numpy(out_grad))
    torch_grad = torch_x.grad.numpy()
    assert np.allclose(torch_grad, grad_out[0].asnumpy(), 0.0001, 0.0001)
