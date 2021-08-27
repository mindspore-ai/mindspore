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
"""
test flatten api
"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import _cell_graph_executor


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()

    def construct(self, input_x):
        return self.flatten(input_x)


def test_compile():
    net = Net()
    input_data = Tensor(np.array([[1.2, 2.1], [2.2, 3.2]]).astype(np.float32))
    _cell_graph_executor.compile(net, input_data)
