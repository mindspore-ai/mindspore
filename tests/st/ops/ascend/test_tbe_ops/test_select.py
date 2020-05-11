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
from mindspore.nn import Cell
import mindspore.context as context
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import mindspore as ms
from mindspore.train.model import Model

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Select(Cell):
    def __init__(self, dtype):
        super(Select, self).__init__()
        self.select = P.Select()

    def construct(self, cond, inputa, inputb):
        return self.select(cond, inputa, inputb)


def me_select(cond, inputa, inputb, dtype=ms.float32):
    net = Select(dtype)
    net.set_train()
    model = Model(net)
    if isinstance(inputa, np.ndarray) == True:
        inputa = Tensor(inputa)
    if isinstance(inputb, np.ndarray) == True:
        inputb = Tensor(inputb)
    if isinstance(cond, np.bool_) == True:
        cond = np.array(cond)

    out = model.predict(Tensor(cond), inputa, inputb)
    return out.asnumpy()


def cmp_select(input_cond, inputa, inputb):
    cond = input_cond > 0.5
    out_me = me_select(cond, inputa, inputb)
    print(input_cond)
    print(cond)
    print(inputa)
    print(inputb)
    print(out_me)


def test_select_2_2():
    input_cond = np.random.rand(2, 2)
    inputa = np.random.randn(2, 2).astype(np.float32)
    inputb = np.random.randn(2, 2).astype(np.float32)
    cmp_select(input_cond, inputa, inputb)
