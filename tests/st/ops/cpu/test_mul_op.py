# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

x = np.random.uniform(-2, 2, (2, 3, 4, 4)).astype(np.float32)
y = np.random.uniform(-2, 2, (1, 1, 1, 1)).astype(np.float32)

context.set_context(device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = P.Mul()
        self.x = Parameter(initializer(Tensor(x), x.shape), name='x3')
        self.y = Parameter(initializer(Tensor(y), y.shape), name='y3')

    @ms_function
    def construct(self):
        return self.mul(self.x, self.y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_Mul():
    mul = Net()
    output = mul()
    print(x)
    print(y)
    print(output)
