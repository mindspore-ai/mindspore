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
from mindspore import context, Tensor
from mindspore.ops import composite as C
import mindspore

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

grad_all = C.GradOperation(get_all=True)
bprop_debug = False

class MulAdd(nn.Cell):
    def construct(self, x, y):
        return 2 * x * x + y * y

    def bprop(self, x, y, out, dout):
        global bprop_debug
        bprop_debug = True
        return dout, 2 * y


def test_custom_bprop():
    mul_add = MulAdd()
    mul_add.bprop_debug = True
    global bprop_debug
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    y = Tensor(np.array([2, 3, 4]).astype(np.int32))
    grad_all(mul_add)(x, y)
    assert bprop_debug
    bprop_debug = False
    mul_add.set_inputs(Tensor(shape=[None], dtype=mindspore.float32), Tensor(shape=[None], dtype=mindspore.float32))
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    y = Tensor(np.array([2, 3, 4]).astype(np.int32))
    grad_all(mul_add)(x, y)
    assert bprop_debug


class Net(nn.Cell):
    def construct(self, x, y):
        return 2 * x * x + y * y

def test_grad_all():
    net = Net()
    x = Tensor(np.array([1, 2, 3]).astype(np.int32))
    y = Tensor(np.array([2, 3, 4]).astype(np.int32))
    res = grad_all(net)(x, y)
    print(res)
