# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test cases for categorical distribution"""

import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.distribution as msd
from mindspore import Tensor
from mindspore import dtype as ms
import pytest

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

def generate_probs(seed, shape):
    np.random.seed(seed)
    probs = np.random.dirichlet(np.ones(shape[3]), size=1)
    for _ in range(shape[0] - 1):
        for _ in range(shape[1] - 1):
            for _ in range(shape[2] - 1):
                probs = np.vstack(((np.random.dirichlet(np.ones(shape[3]), size=1)), probs))
            probs = np.array([probs, probs])
        probs = np.array([probs, probs])
    return probs


class CategoricalProb(nn.Cell):
    def __init__(self, probs, seed=10, dtype=ms.int32, name='Categorical'):
        super().__init__()
        self.b = msd.Categorical(probs, seed, dtype, name)

    def construct(self, value, probs=None):
        out1 = self.b.prob(value, probs)
        out2 = self.b.log_prob(value, probs)
        out3 = self.b.cdf(value, probs)
        out4 = self.b.log_cdf(value, probs)
        out5 = self.b.survival_function(value, probs)
        out6 = self.b.log_survival(value, probs)
        return out1, out2, out3, out4, out5, out6



@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_probability_categorical_prob_cdf_probs_none():
    probs = None
    probs1 = generate_probs(3, shape=(2, 2, 1, 64))
    value = np.random.randint(0, 63, size=(64)).astype(np.float32)
    net = CategoricalProb(probs)
    net(Tensor(value), Tensor(probs1))
