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
import pytest

import mindspore as ms
import mindspore.ops.operations as P
from mindspore import nn, Tensor, context


class Net(nn.Cell):

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.op = P.Shape()

    def construct(self, tensor):
        return self.op(tensor)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_tensor_shape_dyn():
    """
    Feature: test Shape ops in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()

    tensor_dyn = Tensor(shape=[None, None, None], dtype=ms.float32)
    net.set_inputs(tensor_dyn)

    tensor = Tensor(np.ones([3, 2, 1]).astype(np.float32))
    output = net(tensor)

    expect_shape = (3,)
    assert output.asnumpy().shape == expect_shape
