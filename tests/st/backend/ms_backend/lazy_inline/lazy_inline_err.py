# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import lazy_inline
from mindspore.common.parameter import Parameter
from mindspore import context, Tensor
import mindspore.nn as nn


grad = C.GradOperation(get_all=True)


class GradNet(nn.Cell):
    def __init__(self, network):
        super(GradNet, self).__init__()
        self.network = network

    def construct(self, *inputs):
        gout = grad(self.network)(*inputs)
        return gout


class Net(nn.Cell):
    @lazy_inline
    def __init__(self, axis=0, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.addn_input = Parameter(Tensor(np.ones((8, 16, 16)).astype(np.float32)), name="addn")
        self.add_input = Parameter(Tensor(np.ones((8, 16, 16)).astype(np.float32)), name="add")
        self.axis = axis
        self.addn = P.AddN()
        self.add = P.Add()
        self.cumsum = P.CumSum()
        if strategy1 is not None:
            self.addn.shard(strategy1)
        if strategy2 is not None:
            self.add.shard(strategy2)
        if strategy3 is not None:
            self.cumsum.shard(strategy3)

    def construct(self, input_a, label):
        out = self.addn([self.addn_input, input_a])
        out = self.add(out, self.add_input)
        out = self.cumsum(out, self.axis)
        return out


def run_lazy_inline():
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    inputs_ = Tensor(np.random.randn(8, 16, 16).astype(np.float32))
    label_ = Tensor(np.random.randn(1, 1, 1, 1).astype(np.float32))

    grad_net = GradNet(Net())
    grad_net(inputs_, label_)


run_lazy_inline()
