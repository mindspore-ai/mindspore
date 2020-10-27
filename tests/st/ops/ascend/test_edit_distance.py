# Copyright 2020 Huawei Technologies Co., Ltd
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

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")


class EditDistance(nn.Cell):
    def __init__(self, hypothesis_shape, truth_shape, normalize=True):
        super(EditDistance, self).__init__()
        self.edit_distance = P.EditDistance(normalize)
        self.hypothesis_shape = hypothesis_shape
        self.truth_shape = truth_shape

    def construct(self, hypothesis_indices, hypothesis_values, truth_indices, truth_values):
        return self.edit_distance(hypothesis_indices, hypothesis_values, self.hypothesis_shape,
                                  truth_indices, truth_values, self.truth_shape)

def test_edit_distance():
    h1, h2, h3 = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]), np.array([1, 2, 3]), np.array([2, 2, 2])
    t1, t2, t3 = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]), np.array([1, 2, 3, 1]), np.array([2, 2, 2])
    hypothesis_indices = Tensor(h1.astype(np.int64))
    hypothesis_values = Tensor(h2.astype(np.int64))
    hypothesis_shape = Tensor(h3.astype(np.int64))
    truth_indices = Tensor(t1.astype(np.int64))
    truth_values = Tensor(t2.astype(np.int64))
    truth_shape = Tensor(t3.astype(np.int64))
    edit_distance = EditDistance(hypothesis_shape, truth_shape)
    out = edit_distance(hypothesis_indices, hypothesis_values, truth_indices, truth_values)
    print(out)
