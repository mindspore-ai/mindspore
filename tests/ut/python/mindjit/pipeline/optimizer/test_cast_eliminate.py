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
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore import context, ops
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE, save_graphs=True)


def test_input_not_dtype():
    """
    Feature: Two cast elimination.
    Description: The second input of cast is not dtype because the node has not been constantly folded.
    Expectation: Compile successfully
    """

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.y = Parameter(Tensor(np.random.randn(1280, 1280), ms.float32))

        def construct(self, x, tokens):
            _dtype = x.dtype
            indices = ops.stack((ops.arange(x.shape[0]), tokens.argmax(dim=-1)), axis=-1)
            x = ops.gather_nd(x, indices)
            x = ops.matmul(x, ops.cast(ops.cast(self.y, _dtype), x.dtype)).astype(_dtype)
            return x

    x = Tensor(np.random.randn(1, 77, 1280), ms.float16)
    tokens = Tensor(np.random.randn(1, 77), ms.int32)
    net = Net()
    net(x, tokens)
