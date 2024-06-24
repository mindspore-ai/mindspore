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
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, max_length, pad, dtype=mindspore.int32):
        super(Net, self).__init__()
        self.randperm = P.Randperm(max_length, pad, dtype)

    def construct(self, n):
        return self.randperm(n)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: aicpu Randperm
    Description: test Randperm on Acsend
    Expectation: success.
    """
    context.set_context(mode=mode, device_target="Ascend")
    net = Net(max_length=5, pad=-1)
    output = net(Tensor([3], mindspore.int32))

    assert np.all(np.sort(output.asnumpy()) == [-1, -1, 0, 1, 2])
