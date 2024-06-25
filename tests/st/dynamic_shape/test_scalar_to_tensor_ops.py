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

import pytest
import mindspore as ms
from mindspore import nn
from mindspore import ops
from tests.mark_utils import arg_mark


def test_mutable_scalar_input():
    """
    Feature: Operator test.
    Description:  Test ScalarToTensor input is a variable scalar.
    Expectation: A tensor will one element.
    """

    class Net(nn.Cell):
        def construct(self, x):
            out = ops.ScalarToTensor()(x)
            return out

    x = ms.mutable(1.2)
    net = Net()
    out = net(x)
    expect = ms.Tensor(1.2, dtype=ms.float32)
    assert out == expect
