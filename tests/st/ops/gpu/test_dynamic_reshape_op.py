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
import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import _inner_ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.d_reshape = ops.DynamicReshape()

    def construct(self, data, shape):
        return self.d_reshape(data, shape)

@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float32():
    """
    Feature: Dynamci Reshape.
    Description: test cases for dynamicreshape.
    Expectation: the result match expected array.
    """
    data = Tensor(np.arange(1, 9).reshape((2, 4)), mindspore.float32)
    shape = Tensor(np.array([4, 2]), mindspore.int64)
    expect_data = np.arange(1, 9).reshape((4, 2))
    print(data)
    print(shape)
    net = Net()
    output = net(data, shape)
    print(output.asnumpy())
    assert np.array_equal(output.asnumpy(), expect_data)
