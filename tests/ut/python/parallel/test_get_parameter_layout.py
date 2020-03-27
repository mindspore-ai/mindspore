# Copyright 2019 Huawei Technologies Co., Ltd
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

import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter 
import mindspore as ms
import mindspore.common.api as me


def test_get_parameter_layout():
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, weight):
            super().__init__()
            self.weight = Parameter(weight, "w1")
            self.matmul = P.MatMul(transpose_a=False, transpose_b=True).set_strategy(strategy1)
            self.relu = P.ReLU().set_strategy(strategy2)

        def construct(self, x):
            out = self.matmul(x, self.weight)
            out = self.relu(out)
            return out

    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    strategy1 = ((2, 1), (4, 1))
    strategy2 = ((2, 4), )
    context.set_context(mode=context.GRAPH_MODE)

    x = Tensor(np.ones([32, 32]), dtype=ms.float32)
    weight = Tensor(np.ones([64, 32]), dtype=ms.float32)

    net = Net(strategy1, strategy2, weight)
    exe = me._executor
    exe.compile(net, x)
    x_layout = ([2, 4], [1, -1])  # device_arrangement = [2, 4], tensor_map = [1, -1]
    weight_layout = ([2, 4], [0, -1])  # device_arrangement = [2, 4], tensor_map = [0, -1]
    expect_dict = {'x': x_layout, 'w1': weight_layout}
    # to be resovled: static local variable count_p is used in step_parallel.cc, it needs to be reset between each ut 
    assert (net._parameter_layout_dict == expect_dict)


if __name__ == '__main__':
    test_get_parameter_layout()

