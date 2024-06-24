# Copyright 2021 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as ops
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.d_shape = ops.Shape()
        self.d_broadcastto = inner.DynamicBroadcastTo()

    def construct(self, data, shape):
        shape = self.d_shape(shape)
        return self.d_broadcastto(data, shape)

@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_float32():
    """
    Feature: Dynamic BroadcastTo.
    Description: test cases for dynamic_broadcastto.
    Expectation: the result match expected array.
    """
    data = Tensor(np.array([1, 2, 3]), mindspore.float32)
    shape = Tensor(np.zeros((2, 3)), mindspore.int64)
    expect_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32)
    net = Net()
    output = net(data, shape)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), expect_data)
