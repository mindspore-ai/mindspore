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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")

class Net(nn.Cell):
    def __init__(self, perm_in):
        super(Net, self).__init__()
        self.transpose = P.Transpose()
        self.perm = perm_in

    def construct(self, input):
        x = self.transpose(input, self.perm)
        return x

def ms_transpose(input, perm_in):
    context.set_context(mode=context.GRAPH_MODE)
    input_me = Tensor(input)
    net = Net(perm_in)
    net.set_train()
    model = Model(net)
    output = model.predict(input_me)
    print("-------------ms------------------")
    print(output.asnumpy().dtype)
    print(output.asnumpy())

def test_net():
    input = np.random.randn(8, 24, 1, 1).astype(np.float16)
    perm = (0, 2, 3, 1)
    ms_transpose(input, perm)