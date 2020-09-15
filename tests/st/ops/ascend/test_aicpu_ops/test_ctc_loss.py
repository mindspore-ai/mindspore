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

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ctc_loss = P.CTCLoss()

    def construct(self, inputs, labels_indices, labels_values, sequence_length):
        return self.ctc_loss(inputs, labels_indices, labels_values, sequence_length)


def test_net_float32():
    x = np.random.randn(2, 2, 3).astype(np.float32)
    labels_indices = np.array([[0, 1], [1, 0]]).astype(np.int64)
    labels_values = np.array([1, 2]).astype(np.int32)
    sequence_length = np.array([2, 2]).astype(np.int32)
    net = Net()
    output = net(Tensor(x), Tensor(labels_indices), Tensor(labels_values), Tensor(sequence_length))
    print(output)
