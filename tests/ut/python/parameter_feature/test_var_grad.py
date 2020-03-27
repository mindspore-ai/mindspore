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
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
import mindspore.ops.composite as C
from mindspore.common.api import _executor

context.set_context(mode=context.GRAPH_MODE)

def test_net_vargs_expand():
    class AddNet(Cell):
        def __init__(self):
            super(AddNet, self).__init__()
            self.w = Parameter(Tensor(np.ones((3, 4, 5), np.float32)), "w2", requires_grad=True)
        def construct(self, x, y):
            return x + y
    x = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    y = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    sens = Tensor(np.random.normal(0, 1, [3, 4, 5]).astype(np.float32))
    net = AddNet()
    out = C.grad_all_with_sens(net, net.trainable_params())(x, y, sens)
