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
"""
stgcn network with loss.
"""

import mindspore.nn as nn
import mindspore.ops as P

class LossCellWithNetwork(nn.Cell):
    """STGCN loss."""
    def __init__(self, network):
        super(LossCellWithNetwork, self).__init__()
        self.loss = nn.MSELoss()
        self.network = network
        self.reshape = P.Reshape()

    def construct(self, x, label):
        x = self.network(x)
        x = self.reshape(x, (len(x), -1))
        label = self.reshape(label, (len(label), -1))
        STGCN_loss = self.loss(x, label)
        return STGCN_loss
