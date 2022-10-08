# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import pytest
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap
import mindspore.context as context
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self, sample, replacement, seed=0):
        super(Net, self).__init__()
        self.sample = sample
        self.replacement = replacement
        self.seed = seed

    def construct(self, x):
        return C.multinomial(x, self.sample, self.replacement, self.seed)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multinomial():
    x0 = Tensor(np.array([0.9, 0.2]).astype(np.float32))
    x1 = Tensor(np.array([[0.9, 0.2], [0.9, 0.2]]).astype(np.float32))
    net0 = Net(1, True, 20)
    net1 = Net(2, True, 20)
    net2 = Net(6, True, 20)
    out0 = net0(x0)
    out1 = net1(x0)
    out2 = net2(x1)
    assert out0.asnumpy().shape == (1,)
    assert out1.asnumpy().shape == (2,)
    assert out2.asnumpy().shape == (2, 6)


class BatchedMultinomial(nn.Cell):
    def __init__(self):
        super().__init__()
        self.multinomial = P.Multinomial(seed=5, seed2=6)

    def construct(self, prob, num_sample):
        return self.multinomial(prob, num_sample)


def multinomial(prob, num_sample):
    return P.Multinomial(seed=5, seed2=6)(prob, num_sample)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_multinomial_vmap():
    """
    Feature: test Multinomial vmap feature.
    Description: test Multinomial vmap feature.
    Expectation: success.
    """
    prob = Tensor([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], ms.float32)
    num_sample = 3

    batched_multinomial = BatchedMultinomial()
    batched_out = batched_multinomial(prob, num_sample)
    vmap_out = vmap(multinomial, in_axes=(0, None), out_axes=0)(prob, num_sample)

    assert (batched_out.asnumpy() == vmap_out.asnumpy()).all()
