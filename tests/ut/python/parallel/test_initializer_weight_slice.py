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

import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter
import mindspore as ms
import mindspore.common.api as me
from mindspore.common.initializer import initializer
from hccl_test.manage.api import Hccl


def test_initializer_weight_slice():
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

    def get_slice(rank):
        hccl = Hccl()
        rank_save = hccl.rank_id
        hccl.rank_id = rank
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=8, global_rank=0)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        strategy1 = ((2, 1), (4, 1))
        strategy2 = ((2, 4),)
        context.set_context(mode=context.GRAPH_MODE)
        exe = me._executor

        x = Tensor(np.ones([32, 32]), dtype=ms.float32)
        weight = initializer("Uniform", [64, 32], ms.float32)
        net = Net(strategy1, strategy2, weight)
        net.set_auto_parallel()
        exe.compile(net, x, auto_parallel_mode=True, phase='train')
        hccl.rank_id = rank_save
        return net.parameters_dict()['w1'].data.asnumpy()

    slice0 = get_slice(0)
    slice1 = get_slice(1)
    slice4 = get_slice(4)
    slice_shape = slice0.shape

    slice0 = slice0.flatten()
    slice1 = slice1.flatten()
    slice4 = slice4.flatten()
    expect_slice_shape = (16, 32)

    assert expect_slice_shape == slice_shape
    assert all(slice0 == slice4)
    assert any(slice0 != slice1)


if __name__ == '__main__':
    test_initializer_weight_slice()
