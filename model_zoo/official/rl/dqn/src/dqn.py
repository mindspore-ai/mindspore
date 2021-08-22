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
"""DQN net"""

import mindspore.nn as nn
import mindspore.ops as ops


class DQN(nn. Cell):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.linear1 = nn.Dense(input_size, hidden_size)
        self.linear2 = nn.Dense(hidden_size, output_size)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.relu(self.linear1(x))
        return self.linear2(x)


class WithLossCell(nn.Cell):
    """
    network with loss function
    """
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.gather = ops.GatherD()

    def construct(self, x, act, label):
        out = self._backbone(x)
        out = self.gather(out, 1, act)
        loss = self._loss_fn(out, label)
        return loss
