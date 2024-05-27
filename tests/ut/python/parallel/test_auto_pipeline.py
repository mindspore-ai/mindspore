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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import compile_net


class Net(Cell):
    def __init__(self):
        super().__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        out = self.relu(x)
        return out


def test_auto_pipeline():
    """
    Feature: auto pipeline
    Description: test auto pipeline option activation
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="recursive_programming", device_num=16,
                                      global_rank=0)
    context.set_auto_parallel_context(auto_pipeline=True)
    input_x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    net = Net()
    compile_net(net, input_x)
