# Copyright 2022 Huawei Technologies Co., Ltd
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

from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer, Tensor
from mindspore.common import set_seed
set_seed(1)

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout = P.Dropout(0.5)
        self.bias = Parameter(initializer("ones", [1024]), name="bias")
        self.bias_add = P.Add()
        self.add = P.Add()

    def construct(self, input_x, residual):
        output = self.dropout(self.bias_add(input_x, self.bias))
        output = self.add(residual, output[0])
        return output


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bias_dropout_add():
    """"
    Feature: Test bias dropout add fusion
    Description: Test gpu bias dropout add fusion
    Expectation: The results are as expected
    """
    np_x = np.ones((1024, 1024)).astype(np.float32)
    input_x = Tensor(np_x)
    residual = Tensor(np_x)
    net = Net()
    output = net(input_x, residual)
    output_np = output.asnumpy()
    output_sum = np.sum(output_np)
    x_sum = np.sum(np_x)
    assert abs(output_sum - 3 * x_sum) / x_sum < 0.1
