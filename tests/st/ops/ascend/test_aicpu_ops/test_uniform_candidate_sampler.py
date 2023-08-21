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

import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, true_classes, num_true, num_sampled, unique, range_max, seed):
        super(Net, self).__init__()
        self.true_classes = true_classes
        self.num_true = num_true
        self.num_sampled = num_sampled
        self.unique = unique
        self.range_max = range_max
        self.seed = seed
        self.uniformcandidatesampler = P.UniformCandidateSampler(num_true, num_sampled, unique, range_max, seed)

    def construct(self):
        return self.uniformcandidatesampler(self.true_classes)


def test_net():
    """
    Feature: test UniformCandidateSampler op.
    Description: test UniformCandidateSampler op.
    Expectation: success.
    """
    true_classes = Tensor(np.array([[1], [3], [4], [6], [3]]), dtype=mstype.int64)
    num_true = 1
    num_sampled = 3
    unique = False
    range_max = 4
    seed = 1
    net = Net(true_classes, num_true, num_sampled, unique, range_max, seed)
    output1, output2, output3 = net()
    assert output1.shape == (3,)
    assert output2.shape == (5, 1)
    assert output3.shape == (3,)
