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

"""test NLLLoss forward and backward dynamic shape"""

import pytest
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C

context.set_context(mode=context.GRAPH_MODE, device_target="CPU", save_graphs=True)


class NLLLoss(nn.Cell):
    def __init__(self, reduction="none"):
        super().__init__()
        self.nllloss = P.NLLLoss(reduction=reduction)

    def construct(self, x, t, w):
        return self.nllloss(x, t, w)


class NLLLossGrad(nn.Cell):
    def __init__(self, forward, sens):
        super().__init__()
        self.forward = forward
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.sens = sens

    def construct(self, x, t, w):
        return self.grad(self.forward)(x, t, w, self.sens)


np_type = np.float32
logits = Tensor(np.array([[-1.3739, -2.2700, -3.2333, -2.4589, -0.6566],
                          [-1.2156, -2.6026, -1.2200, -1.8731, -1.7119],
                          [-0.7130, -3.3672, -1.5368, -1.8289, -2.3058]]).astype(np_type))
target = Tensor(np.array([1, 0, 4]).astype(np.int32))
weight = Tensor(np.array([0.2, 0.3, 0.1, 0.15, 0.25]).astype(np_type))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nllloss_cpu_none_dynamic_shape():
    """
    Feature: test nllloss op with reduction none.
    Description: test the ops in dynamic shape.
    Expectation: expect correct output shape.
    """
    nllloss = NLLLoss("none")
    logits_dyn = Tensor(shape=[None]*len(logits.shape), dtype=logits.dtype)
    target_dyn = Tensor(shape=[None]*len(target.shape), dtype=target.dtype)
    weight_dyn = Tensor(shape=[None]*len(weight.shape), dtype=weight.dtype)
    nllloss.set_inputs(logits_dyn, target_dyn, weight_dyn)
    loss, total_weight = nllloss(logits, target, weight)
    assert loss.asnumpy().shape == (logits.shape[0],)
    assert total_weight.asnumpy().shape == tuple()

    nllloss_grad = NLLLossGrad(nllloss, sens=(loss + 0.5, total_weight + 0.5))
    nllloss_grad.set_inputs(logits_dyn, target_dyn, weight_dyn)
    expect_grad = nllloss_grad(logits, target, weight)
    assert expect_grad[0].asnumpy().shape == logits.asnumpy().shape
    assert expect_grad[1].asnumpy().shape == target.asnumpy().shape
    assert expect_grad[2].asnumpy().shape == weight.asnumpy().shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_nllloss_cpu_mean_dynamic_shape():
    """
    Feature: test nllloss op with reduction mean.
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    nllloss = NLLLoss("mean")
    logits_dyn = Tensor(shape=[None]*len(logits.shape), dtype=logits.dtype)
    target_dyn = Tensor(shape=[None]*len(target.shape), dtype=target.dtype)
    weight_dyn = Tensor(shape=[None]*len(weight.shape), dtype=weight.dtype)
    nllloss.set_inputs(logits_dyn, target_dyn, weight_dyn)
    loss, total_weight = nllloss(logits, target, weight)
    assert loss.asnumpy().shape == tuple()
    assert total_weight.asnumpy().shape == tuple()

    nllloss_grad = NLLLossGrad(nllloss, sens=(loss + 0.5, total_weight + 0.5))
    nllloss_grad.set_inputs(logits_dyn, target_dyn, weight_dyn)
    expect_grad = nllloss_grad(logits, target, weight)
    assert expect_grad[0].asnumpy().shape == logits.asnumpy().shape
    assert expect_grad[1].asnumpy().shape == target.asnumpy().shape
    assert expect_grad[2].asnumpy().shape == weight.asnumpy().shape
