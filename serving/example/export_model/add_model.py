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
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.train.serialization import export

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.TensorAdd()

    def construct(self, x_, y_):
        return self.add(x_, y_)

x = np.ones(4).astype(np.float32)
y = np.ones(4).astype(np.float32)

def export_net():
    add = Net()
    output = add(Tensor(x), Tensor(y))
    export(add, Tensor(x), Tensor(y), file_name='tensor_add.mindir', file_format='MINDIR')
    print(x)
    print(y)
    print(output.asnumpy())

if __name__ == "__main__":
    export_net()
    