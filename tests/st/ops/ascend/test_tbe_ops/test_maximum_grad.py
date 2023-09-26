# Copyright 2020-2023 Huawei Technologies Co., Ltd
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

import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")
grad = C.GradOperation(get_all=True, sens_param=True)


class MaxNetMe(Cell):
    def __init__(self):
        super(MaxNetMe, self).__init__()
        self.max = P.Maximum()

    def construct(self, inputA, inputB):
        x = self.max(inputA, inputB)
        return x


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, inputA, inputB, sens):
        gout = grad(self.network)(inputA, inputB, sens)
        return gout


def gen_data(inputA_np, inputB_np, grad_=None):
    inputA_me = inputA_np
    if isinstance(inputA_np, np.ndarray):
        inputA_me = Tensor(inputA_me)
    inputB_me = inputB_np
    if isinstance(inputB_np, np.ndarray):
        inputB_me = Tensor(inputB_np)
    if grad_ is None:
        grad_ = np.random.randn(1, 3, 2, 2).astype(np.float32)

    net_me = GradWrap(MaxNetMe())
    net_me.set_train()
    output_me = net_me(inputA_me, inputB_me, Tensor(grad_))
    ms_inputA_grad = output_me[0].asnumpy()
    ms_inputB_grad = output_me[1].asnumpy()

    torch_inputA = torch.tensor(inputA_np.astype(np.float32), requires_grad=True)
    torch_inputB = torch.tensor(inputB_np.astype(np.float32), requires_grad=True)
    output_pt = torch.max(torch_inputA, torch_inputB)
    grad_pt = torch.from_numpy(grad_.astype(np.float32))
    output_pt.backward(grad_pt)
    torch_inputA_grad = torch_inputA.grad.detach().numpy()
    torch_inputB_grad = torch_inputB.grad.detach().numpy()

    assert np.allclose(ms_inputA_grad, torch_inputA_grad, rtol=1e-6, atol=1e-4)
    assert np.allclose(ms_inputB_grad, torch_inputB_grad, rtol=1e-6, atol=1e-4)


def test_net():
    """
    Feature: test maximum grad on ascend
    Description: test maximumgrad with 4D input.
    Expectation: result match to torch result.
    """
    inputA_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    inputB_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    gen_data(inputA_np, inputB_np)


def test_max_tensor_grad_with_same_input():
    """
    Feature: test maximum grad on ascend
    Description: test maximumgrad with same input.
    Expectation: result match to torch result.
    """
    inputA_np = np.array([1.8, 5.6, 9.3]).astype(np.float32)
    inputB_np = np.array([2.3, 5.6, 5.8]).astype(np.float32)
    grad_ = np.array([1.0, -1.0, 0]).astype(np.float32)
    gen_data(inputA_np, inputB_np, grad_)
