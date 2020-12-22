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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    """Net definition"""

    def __init__(self,
                 output_channels,
                 bias_init='zeros',
                 ):
        super(Net, self).__init__()
        self.biasAdd = P.BiasAdd()

        if isinstance(bias_init, Tensor):
            if bias_init.ndim != 1 or bias_init.shape[0] != output_channels:
                raise ValueError("bias_init shape error")

        self.bias = Parameter(initializer(
            bias_init, [output_channels]), name="bias")

    def construct(self, input_x):
        return self.biasAdd(input_x, self.bias)


def test_compile():
    bias_init = Tensor(np.ones([3]).astype(np.float32))
    net = Net(3, bias_init=bias_init)
    input_data = Tensor(np.ones([1, 3, 3, 3], np.float32))
    # since simulator currently not support matMul
    # enable it when staging function is ready
    output = net(input_data)
    print(output.asnumpy())
