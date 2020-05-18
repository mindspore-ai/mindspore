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
import os

import mindspore as ms
import mindspore.communication.management as distributedTool
from mindspore import context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.nn import Dropout

device_num = 4
device_id = int(os.environ["RANK_ID"])
path = "./output/"


def setup_module():
    print("~~~~~~~~~~~set up~~~~~~~~~~~~~")
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(device_num=device_num, global_rank=device_id)
    distributedTool.init()
    distributedTool.create_group("0-3", [0, 1, 2, 3])
    print("~~~~~~~~~~~set up finished~~~~~~~~~~~~~")


def teardown_module():
    print("~~~~~~~~~~~~tear down~~~~~~~~~~")


class Net(Cell):
    def __init__(self, keep_prob, seed0, seed1, strategy=None):
        super(Net, self).__init__()
        self.drop = Dropout(keep_prob, seed0, seed1, dtype=ms.float32, strategy=strategy)

    def construct(self, input):
        x = self.drop(input)
        return x


class DropoutFactory:
    def __init__(self, input_shape, keep_prob, seed0, seed1, strategy0=None):
        size = 1
        prefix = ""
        for s in input_shape:
            prefix = prefix + str(s)
            size = size * s
        self.prefix = prefix
        number_range = min(10, size)
        self.input_np = np.reshape(np.arange(0, size) % number_range, input_shape).astype(np.float32)
        self.keep_prob = keep_prob
        self.seed0 = seed0
        self.seed1 = seed1
        self.strategy0 = strategy0
        need_dev_num = 1
        for s in strategy0[1]:
            need_dev_num = need_dev_num * s
        self.x_id = device_id % need_dev_num
        self.out_id = device_id % need_dev_num

    def get_parallel_blocks(self, input_, strategy):
        blocks = [input_]
        i = 0
        for stra in strategy:
            temp = []
            while len(blocks) > 0:
                block = blocks.pop(0)
                temp.extend(np.split(block, stra, axis=i))
            blocks.extend(temp)
            i += 1
        return blocks

    def d4_tensor_compare(self, input, out_me):
        [a, b, c, d] = input.shape
        for i in range(a):
            for j in range(b):
                for k in range(c):
                    for e in range(d):
                        if out_me[i, j, k, e] == 0:
                            assert True == True
                        else:
                            assert np.allclose(out_me[i, j, k, e], input[i, j, k, e] * (1 / 0.4), 0.0001, 0.0001)

    def forward_mindspore_parallel_impl(self):
        x = Tensor(self.input_np)
        inputs_x = self.get_parallel_blocks(self.input_np, self.strategy0[1])
        x1 = Tensor(inputs_x[self.x_id])
        net = Net(0.4, 0, 0, strategy=self.strategy0)
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
        net.set_auto_parallel()
        out = net(x, parallel_inputs_compile=[x], parallel_inputs_run=[x1])
        return out.asnumpy()

    def forward_cmp(self):
        out_mindspore_parallel = self.forward_mindspore_parallel_impl()
        input_blocks = self.get_parallel_blocks(self.input_np, self.strategy0[1])
        self.d4_tensor_compare(input_blocks[self.out_id], out_mindspore_parallel)


def test_reid_dropout_forward_seed_F32_64_512_8_8():
    fact = DropoutFactory(input_shape=(64, 512, 8, 8), keep_prob=0.4, seed0=0, seed1=0, strategy0=(0, (4, 1, 1, 1)))
    fact.forward_cmp()


def test_reid_dropout_forward_seed_F32_64_512_8_8_repeat():
    fact = DropoutFactory(input_shape=(64, 512, 8, 8), keep_prob=0.4, seed0=0, seed1=0, strategy0=(0, (2, 1, 1, 1)))
    fact.forward_cmp()
