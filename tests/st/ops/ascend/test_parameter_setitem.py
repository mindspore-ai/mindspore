# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import pytest
import numpy as np
import mindspore
from mindspore import nn, context, Tensor, Parameter


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.weight = Parameter(Tensor(np.zeros((20, 48)), dtype=mindspore.float32), name='weight')

    def construct(self, index, value):
        self.weight[index] = value
        return True


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_parameter_setitem():
    """
    Feature: Apply setitem on `Parameter` on graph mode.
    Description: Apply setitem on `Parameter` on graph mode.
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    # setitem by tensor with scalar
    net(Tensor(np.arange(9)), 1)
    # setitem by int with scalar
    net(9, 1)
    # setitem by tensor with tensor
    net(Tensor(np.arange(10, 20)), Tensor(np.ones((10, 48))))
    assert np.allclose(net.weight.asnumpy(), np.ones(net.weight.shape))
