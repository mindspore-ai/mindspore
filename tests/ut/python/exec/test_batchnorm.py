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
"""ut for batchnorm layer"""
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
from ..ut_filter import non_graph_engine


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=3, eps=1e-5, momentum=0.1)

    def construct(self, input_x):
        return self.bn(input_x)


@non_graph_engine
def test_compile():
    net = Net()
    input_data = Tensor(np.ones([1, 3, 4, 4]).astype(np.float32))
    output = net(input_data)
    print(input_data)
    print(output.asnumpy())
