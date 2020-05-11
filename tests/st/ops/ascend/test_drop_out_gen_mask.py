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
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import numpy as np
import mindspore.context as context

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mask = P.DropoutGenMask(10, 28)
        self.shape = P.Shape()

    def construct(self, x, y):
        shape_x = self.shape(x)
        return self.mask(shape_x, y)


x = np.ones([2, 4, 2, 2]).astype(np.int32)
y = np.array([1.0]).astype(np.float32)


def test_net():
    mask = Net()
    tx, ty = Tensor(x), Tensor(y)
    output = mask(tx, ty)
    print(output.asnumpy())
    assert ([255, 255, 255, 255] == output.asnumpy()).all()
