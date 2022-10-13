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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetEqualCount(nn.Cell):
    def __init__(self):
        super(NetEqualCount, self).__init__()
        self.equalcount = P.EqualCount()

    def construct(self, x, y):
        return self.equalcount(x, y)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_equalcount():
    x = Tensor(np.array([1, 20, 5]).astype(np.int32))
    y = Tensor(np.array([2, 20, 5]).astype(np.int32))
    expect = np.array([2]).astype(np.int32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    equal_count = NetEqualCount()
    output = equal_count(x, y)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal_count = NetEqualCount()
    output = equal_count(x, y)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_equalcount_dynamic():
    """
    Feature: EqualCount ops
    Description: dynamic shape in pynative and graph mode in GPU
    Expectation: success
    """
    x = Tensor(np.array([1, 20, 5]).astype(np.int32))
    y = Tensor(np.array([2, 20, 5]).astype(np.int32))
    xx = Tensor(shape=[None], dtype=mindspore.int32)
    yy = Tensor(shape=[None], dtype=mindspore.int32)
    expect = np.array([2]).astype(np.int32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    equal_count = NetEqualCount()
    equal_count.set_inputs(xx, yy)
    output = equal_count(x, y)
    assert (output.asnumpy() == expect).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    equal_count = NetEqualCount()
    equal_count.set_inputs(xx, yy)
    output = equal_count(x, y)
    assert (output.asnumpy() == expect).all()
