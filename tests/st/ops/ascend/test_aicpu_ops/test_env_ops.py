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
""" test grad ops """
import mindspore.ops as ops
from mindspore import ms_function
from mindspore import Tensor, context
from mindspore.common import dtype as mstype

one = Tensor([1], mstype.int32)
zero = Tensor([0], mstype.int32)

@ms_function
def local_pow(x, n):
    r = one
    while n > zero:
        n = n - one
        r = r * x
    return r

def test_pow_first_order():
    """
    Feature: pow first order test.
    Description: pow first order test.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    n = Tensor([3], mstype.int32)
    grad = ops.GradOperation()
    grad_net = grad(local_pow)
    res = grad_net(x, n)
    assert res == 75

def test_pow_second_order():
    """
    Feature: pow second order test.
    Description: pow second order test.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = Tensor([5], mstype.int32)
    n = Tensor([3], mstype.int32)
    grad = ops.GradOperation()
    grad_net = grad(local_pow)
    sec_grad_net = grad(grad_net)
    res = sec_grad_net(x, n)
    assert res == 30
