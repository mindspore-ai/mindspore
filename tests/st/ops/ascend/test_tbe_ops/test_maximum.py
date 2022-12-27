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

context.set_context(device_target="Ascend")


class Max(nn.Cell):
    def __init__(self, dtype):
        super(Max, self).__init__()
        self.max = P.Maximum()

    def construct(self, inputa, inputb):
        return self.max(inputa, inputb)


def me_max(inputa, inputb, dtype=ms.float32):
    context.set_context(mode=context.GRAPH_MODE)
    net = Max(dtype)
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


def cmp_max(a, b):
    out = np.maximum(a, b)
    out_ms = me_max(a, b)
    print("-------ms------")
    print("numpy out :{}".format(out))
    print("ms out :{}".format(out_ms))


def test_maximum_2_2():
    a = np.random.randn(2, 2).astype(np.float32)
    b = np.random.randn(2, 2).astype(np.float32)
    cmp_max(a, b)
