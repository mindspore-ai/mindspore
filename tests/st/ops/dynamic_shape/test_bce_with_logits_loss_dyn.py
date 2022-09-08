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

import math
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetDyn(nn.Cell):
    def __init__(self, reduction, indices):
        super(NetDyn, self).__init__()
        self.indices = indices
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.loss = P.BCEWithLogitsLoss(reduction=reduction)

    def construct(self, predict, target, weight, pos_weight):
        unique_indice, _ = self.unique(self.indices)
        predict = self.gather(predict, unique_indice, 0)
        return self.loss(predict, target, weight, pos_weight)


def test_bce_mean_dyn_ascend():
    """
    Feature: Test dynamic shape of BCEWithLogitsLoss op that the reduction is mean on ascend.
    Description:  The shape of input is dynamic.
    Expectation: Assert that results are consistent with expect.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    predict = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    target = Tensor(np.arange(34, 40).reshape(2, 3).astype(np.float32))
    weight = Tensor(np.array([2, 3, 1]).astype(np.float32))
    pos_weight = Tensor(np.array([6, 3, 4]).astype(np.float32))
    indices = Tensor(np.array([0, 1]))
    loss = NetDyn("mean", indices)
    output = loss(predict, target, weight, pos_weight)
    expected = -113.55404
    # assert scalar
    assert math.isclose(output.asnumpy().tolist(), expected, rel_tol=1e-4, abs_tol=1e-4)
