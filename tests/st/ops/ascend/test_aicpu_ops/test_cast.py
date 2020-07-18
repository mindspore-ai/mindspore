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
import mindspore.common.dtype as mstype
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self, x, dtype):
        super(Net, self).__init__()
        self.cast = P.Cast()
        self.x = x
        self.dtype = dtype

    def construct(self):
        return self.cast(self.x, self.dtype)

def test_net_f32_bool():
    x = np.random.randn(3, 4).astype(np.float32)
    x[:, 1] = 0
    net = Net(Tensor(x), mstype.bool_)
    output = net()
    print(output.asnumpy())
    print(Tensor(x).dtype)
    print(output.dtype)

def test_net_f16_bool():
    x = np.random.randn(3, 4).astype(np.float16)
    x[:, 1] = 0
    net = Net(Tensor(x), mstype.bool_)
    output = net()
    print(output.asnumpy())
    print(Tensor(x).dtype)
    print(output.dtype)

def test_net_f64_bool():
    x = np.random.randn(3, 4).astype(np.float64)
    x[:, 1] = 0
    net = Net(Tensor(x), mstype.bool_)
    output = net()
    print(output.asnumpy())
    print(Tensor(x).dtype)
    print(output.dtype)

def test_net_int16_float16():
    x = np.random.randint(-512, 512, size=(3, 4)).astype(np.int16)
    net = Net(Tensor(x), mstype.float16)
    output = net()
    print(output.asnumpy())
    print(Tensor(x).dtype)
    print(output.dtype)

def test_net_int64_float16():
    x = np.random.randint(-512, 512, size=(3, 4)).astype(np.int64)
    net = Net(Tensor(x), mstype.float16)
    output = net()
    print(output.asnumpy())
    print(Tensor(x).dtype)
    print(output.dtype)
