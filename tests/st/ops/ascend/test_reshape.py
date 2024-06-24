# Copyright 2019 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.reshape = P.Reshape()

    def construct(self, tensor, shape):
        return self.reshape(tensor, shape)


def test_net():
    x = np.random.randn(1, 16, 1, 1).astype(np.float16)
    reshape = Net()
    output = reshape(Tensor(x), (1, 16))
    print(output.asnumpy())


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_reshape_bfloat16(mode):
    """
    Feature: test Reshape forward.
    Description: test bfloat16 inputs.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor(np.random.randn(4096, 1536), mstype.bfloat16)
    shape = (1, 4096, 12, 128)
    reshape = Net()
    output = reshape(x, shape)
    assert output.shape == shape
