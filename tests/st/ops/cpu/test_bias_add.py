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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add = P.BiasAdd()

    def construct(self, x, b):
        return self.bias_add(x, b)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add4d():
    x = np.ones([2, 3, 4, 4]).astype(np.float32)
    b = np.array([1, 1, 1]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = np.ones([2, 3, 4, 4]).astype(np.float32) * 2
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add2d():
    x = np.ones([2, 3]).astype(np.float32)
    b = np.array([1, 1, 1]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = np.ones([2, 3]).astype(np.float32) * 2
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add3d():
    x = np.ones([2, 3, 4]).astype(np.float32)
    b = np.array([1, 1, 1]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = np.ones([2, 3, 4]).astype(np.float32) * 2
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add5d():
    x = np.ones([2, 5, 4, 4, 4]).astype(np.float32)
    b = np.array([1, 1, 1, 1, 1]).astype(np.float32)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b))
    expect_output = np.ones([2, 5, 4, 4, 4]).astype(np.float32) * 2
    print(output)
    assert np.all(output.asnumpy() == expect_output), "bias_add execute failed, please check current code commit"
