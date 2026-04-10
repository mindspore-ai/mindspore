# Copyright 2023 Huawei Technologies Co., Ltd
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
"""run no grad test"""
import pytest
from .share.utils import match_array, pi_jit_with_config
from tests.mark_utils import arg_mark
import mindspore
from mindspore import nn, ops, jit, Tensor, _no_grad, context, Parameter
from tests.st.pi_jit.share.utils import pi_jit_with_config


class GradNet(nn.Cell):
    def __init__(self):
        super(GradNet, self).__init__()
        self.w = Parameter(Tensor([5.0], mindspore.float32), name='w')
        self.b = Parameter(Tensor([5.0], mindspore.float32), name='b')

    def construct(self, x):
        y = self.w * x + self.b
        with _no_grad():
            m = y * self.w
        z = m * y
        return z

class GradNetJit(nn.Cell):
    def __init__(self):
        super(GradNetJit, self).__init__()
        self.w = Parameter(Tensor([5.0], mindspore.float32), name='w')
        self.b = Parameter(Tensor([5.0], mindspore.float32), name='b')

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def construct(self, x):
        y = self.w * x + self.b
        with _no_grad():
            m = y * self.w
        z = m * y
        return z

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('input', [Tensor([2], mindspore.float32)])
def test_network_func(input):
    """
    Feature: integrate no grad function into graph
    Description: replace no grad into StopGradient
    Expectation: no error
    TEST_SUMMARY: match the result with pynative
    """
    model_py = GradNet()

    @pi_jit_with_config(jit_config={"compile_with_try": False})
    def test_network(x):
        grad_fn = ops.grad(model_py)
        gradients = grad_fn(x)
        return gradients

    context.set_context(mode=context.PYNATIVE_MODE)
    gradient_jit = test_network(input)
    gradient_py = ops.grad(model_py)(input)
    match_array(gradient_jit, gradient_py, error=0, err_msg=str(gradient_jit))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard')
@pytest.mark.parametrize('input', [Tensor([2], mindspore.float32)])
def test_network_jit_func(input):
    """
    Feature: integrate no grad function into graph
    Description: replace no grad into StopGradient
    Expectation: no error
    TEST_SUMMARY: match the result with pynative
    """
    model_py = GradNetJit()

    def test_network(x):
        grad_fn = ops.grad(model_py)
        gradients = grad_fn(x)
        return gradients

    context.set_context(mode=context.PYNATIVE_MODE)
    gradient_jit = test_network(input)
    gradient_py = ops.grad(model_py)(input)
    match_array(gradient_jit, gradient_py, error=0, err_msg=str(gradient_jit))