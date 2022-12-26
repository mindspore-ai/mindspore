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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.train import Model

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Min(nn.Cell):
    def __init__(self, dtype):
        super(Min, self).__init__()
        self.min = P.Minimum()

    def construct(self, inputa, inputb):
        return self.min(inputa, inputb)


def me_min(inputa, inputb, dtype=ms.float32):
    context.set_context(mode=context.GRAPH_MODE)
    net = Min(dtype)
    net.set_train()
    model = Model(net)
    print(type(inputa))
    if isinstance(inputa, np.ndarray):
        inputa = Tensor(inputa)
    if isinstance(inputb, np.ndarray):
        inputb = Tensor(inputb)
    out = model.predict(inputa, inputb)
    print(out)
    return out.asnumpy()


def cmp_min(a, b):
    print(a)
    print(b)

    out = np.minimum(a, b)
    print(out)
    out_me = me_min(a, b)
    print(out_me)


def test_minimum_2_2():
    a = np.random.randn(2, 2, 1, 1).astype(np.float32)
    b = np.random.randn(2, 2, 1, 1).astype(np.float32)
    cmp_min(a, b)
