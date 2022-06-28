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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class NetUniqueWithPad(nn.Cell):
    def __init__(self):
        super(NetUniqueWithPad, self).__init__()
        self.unique = P.UniqueWithPad()

    def construct(self, x, pad_num):
        x_unique, x_idx = self.unique(x, pad_num)
        return x_unique, x_idx


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_int32():
    """
    Feature: test uniquewithpad in gpu.
    Description: test uniquewithpad forward with int32 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 5]).astype(np.int32))
    exp_output = np.array([1, 2, 3, 4, 5, 99, 99, 99]).astype(np.int32)
    exp_idx = np.array([0, 1, 1, 2, 2, 2, 3, 4]).astype(np.int32)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUniqueWithPad()
    x_unique, x_idx = net(x, 99)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_unique_with_pad_int64():
    """
    Feature: test uniquewithpad in gpu.
    Description: test uniquewithpad forward with int64 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 2, 3, 3, 3, 4, 5]).astype(np.int64))
    exp_output = np.array([1, 2, 3, 4, 5, 99, 99, 99]).astype(np.int64)
    exp_idx = np.array([0, 1, 1, 2, 2, 2, 3, 4]).astype(np.int64)
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetUniqueWithPad()
    x_unique, x_idx = net(x, 99)
    assert (x_unique.asnumpy() == exp_output).all()
    assert (x_idx.asnumpy() == exp_idx).all()
