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
from mindspore.common.api import ms_function
import numpy as np
import mindspore.context as context

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.less = P.Less()

    @ms_function
    def construct(self, x1, x2):
        return self.less(x1, x2)


x1 = np.random.randn(3, 4).astype(np.float16)
x2 = np.random.randn(3, 4).astype(np.float16)


def test_net():
    less = Net()
    output = less(Tensor(x1), Tensor(x2))
    print(x1)
    print(x2)
    print(output.asnumpy())
