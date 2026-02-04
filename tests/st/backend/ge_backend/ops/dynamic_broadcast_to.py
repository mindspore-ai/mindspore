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
import mindspore.context as context
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.operations as ops
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as inner


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.d_shape = ops.Shape()
        self.d_broadcastto = inner.DynamicBroadcastTo()

    def construct(self, data, shape):
        shape = self.d_shape(shape)
        return self.d_broadcastto(data, shape)


def test_dynamic_broadcast_to():
    """
    Feature: for DynamicBroadcastTo op
    Description: inputs are data and shape
    Expectation: the result is correct
    """
    data = Tensor(np.array([1, 2, 3]), mstype.float32)
    shape = Tensor(np.zeros((2, 3)), mstype.int64)
    expect_data = np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32)
    net = Net()
    output = net(data, shape)
    assert np.array_equal(output.asnumpy(), expect_data)


if __name__ == "__main__":
    test_dynamic_broadcast_to()
