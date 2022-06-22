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
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.array_ops import TensorScatterElements

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.scatter_elements = TensorScatterElements(0)

    def construct(self, data, indices, updates):
        return self.scatter_elements(data, indices, updates)


def test_net():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
    indices = np.array([[1, 0, 2], [0, 2, 1]]).astype(np.int32)
    updates = np.array([[0, 0, 0], [0, 0, 0]]).astype(np.float32)
    net = Net()
    tdata = Tensor(data)
    tindices = Tensor(indices)
    tupdates = Tensor(updates)
    output = net(tdata, tindices, tupdates)
    print(output.asnumpy())
    assert np.all([[0.0, 0.0, 3.0], [0.0, 5.0, 0.0], [7.0, 0.0, 0.0]] == output.asnumpy())
