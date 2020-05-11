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
import mindspore.nn as nn
import numpy as np
import mindspore.context as context
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops.operations import Minimum

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
grad = C.GradOperation('get_all', get_all=True, sens_param=True)


class MinNetMe(Cell):
    def __init__(self):
        super(MinNetMe, self).__init__()
        self.min = Minimum()

    def construct(self, inputA, inputB):
        x = self.min(inputA, inputB)
        return x


class GradWrap(Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, inputA, inputB, sens):
        gout = grad(self.network)(inputA, inputB, sens)
        return gout


def gen_data(inputA_np, inputB_np, grad=None):
    inputA_me = inputA_np
    if isinstance(inputA_np, np.ndarray) == True:
        inputA_me = Tensor(inputA_me)

    inputB_me = inputB_np
    if isinstance(inputB_np, np.ndarray) == True:
        inputB_me = Tensor(inputB_np)

    if grad is None:
        grad = np.random.randn(1, 3, 2, 2).astype(np.float32)

    print(inputA_np)
    print(inputB_np)
    print(grad)

    net_me = GradWrap(MinNetMe())
    net_me.set_train()
    output = net_me(inputA_me, inputB_me, Tensor(grad))
    print(output[0].asnumpy())
    print(output[1].asnumpy())


def test_min_tensor_grad_4d():
    inputA_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    inputB_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    gen_data(inputA_np, inputB_np)
