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
""" test_int64_support """
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore as ms


def test_parser_support_int64_normal_graph():
    """ test tensor index support int64 -index, graph mode"""
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()

        def construct(self, inputs, tensor_in):
            result = inputs[tensor_in]
            return result

    context.set_context(mode=context.GRAPH_MODE)
    input_np_x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x, ms.float32)
    input_np_y = np.random.randint(2, size=[1, 2]).astype(np.int64)
    tensor = Tensor(input_np_y, ms.int64)
    net = Net()
    net(input_me_x, tensor).asnumpy()
