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
from tests.mark_utils import arg_mark
import numpy as np
from mindspore import Tensor
from mindspore import context
import mindspore.nn as nn
import pytest

context.set_context(mode=context.GRAPH_MODE)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.x = Tensor([1.1, 2, 3, -4.1, 0, 3, 2, 0]).astype("float32")

    def construct(self):
        return self.x.clip(-5, 5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_clip():
    r"""
    Feature: Functionality of tensor.clip()
    Description: Test case of tensor.clip().
    Expectation: success
    """
    net = Net()
    out = net()
    expected_output = Tensor(np.array([1.1, 2, 3, -4.1, 0, 3, 2, 0]).astype(np.float32))
    assert np.array_equal(out, expected_output)
