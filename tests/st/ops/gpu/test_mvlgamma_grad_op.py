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
import torch
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations._grad_ops as P
from mindspore import Tensor
from mindspore.common.api import ms_function

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class MvlgammaGradNet(nn.Cell):
    def __init__(self, nptype, p):
        super(MvlgammaGradNet, self).__init__()
        self.mvlgamma_grad = P.MvlgammaGrad(p=p)
        self.y_grad_np = np.array([[3, 4, 5], [4, 2, 6]]).astype(nptype)
        self.y_grad = Tensor(self.y_grad_np)
        self.x_np = np.array([[3, 4, 5], [4, 2, 6]]).astype(nptype)
        self.x = Tensor(self.x_np)


    @ms_function
    def construct(self):
        return self.mvlgamma_grad(self.y_grad, self.x)


def mvlgamma_grad_torch(y_grad_np, x_np, p):
    x_torch = torch.tensor(x_np, requires_grad=True)
    grad_torch = torch.tensor(y_grad_np)
    out_torch = torch.mvlgamma(x_torch, p=p)
    out_torch.backward(grad_torch)
    dx = x_torch.grad
    return dx.numpy()


def mvlgamma_grad(nptype, p):
    mvlgamma_ = MvlgammaGradNet(nptype, p)
    mvlgamma_output = mvlgamma_().asnumpy()
    mvlgamma_expect = mvlgamma_grad_torch(mvlgamma_.y_grad_np, mvlgamma_.x_np, p).astype(nptype)
    assert np.allclose(mvlgamma_output, mvlgamma_expect, 1e-4, 1e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mvlgamma_graph_float32():
    """
    Feature: ALL To ALL
    Description: test cases for MvlgammaGrad
    Expectation: the result match to numpy
    """
    mvlgamma_grad(np.float32, 2)
