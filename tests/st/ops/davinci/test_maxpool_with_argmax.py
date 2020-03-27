# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.common.api import ms_function
import numpy as np
import mindspore.context as context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
context.set_context(device_target="Ascend")
class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()

        self.maxpool = P.MaxPoolWithArgmax(pad_mode="same",
                             window=3,
                             stride=2)
        self.x = Parameter(initializer(
                           'normal', [1, 64, 112, 112]), name='w')
        self.add = P.TensorAdd()


    @ms_function
    def construct(self):
        output = self.maxpool(self.x)
        return output[0]

def test_net():
    x = np.random.randn(1,64,112,112).astype(np.float32)
    maxpool = Net()
    output = maxpool()
    print("***********output output*********")
    print(output.asnumpy())
