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
""" test clip_by_norm """
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ..ut_filter import non_graph_engine


@non_graph_engine
def test_clip_by_norm():
    clip_by_norm = nn.ClipByNorm()
    x = Tensor(np.array([[-2, 0, 0], [0, 3, 4]]).astype(np.float32))
    clip_norm = Tensor(np.array([1]).astype(np.float32))
    clip_by_norm(x, clip_norm)


@non_graph_engine
def test_clip_by_norm_const():
    class Network(nn.Cell):
        def __init__(self):
            super(Network, self).__init__()
            self.norm_value = Tensor(np.array([1]).astype(np.float32))
            self.clip = nn.ClipByNorm()

        def construct(self, x):
            return self.clip(x, self.norm_value)

    net = Network()
    x = Tensor(np.array([[-2, 0, 0], [0, 3, 4]]).astype(np.float32))
    net(x)
