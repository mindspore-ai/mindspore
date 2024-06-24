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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype

context.set_context(device_target="Ascend")


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.argmax = P.Argmax(axis=1)

    @jit
    def construct(self, x):
        return self.argmax(x)


def test_net():
    x = np.random.randn(32, 10).astype(np.float32)
    argmax = Net()
    output = argmax(Tensor(x))
    print(x)
    print(output.asnumpy())


class ArgmaxFuncNet(nn.Cell):
    def construct(self, x):
        return F.argmax(x, dim=-1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_functional_argmax(mode):
    """
    Feature: Test argmax functional api.
    Description: Test argmax functional api for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="Ascend")
    x = Tensor([[1, 20, 5], [67, 8, 9], [130, 24, 15]], mstype.float32)
    net = ArgmaxFuncNet()
    output = net(x)
    expect_output = np.array([1, 0, 0]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)
