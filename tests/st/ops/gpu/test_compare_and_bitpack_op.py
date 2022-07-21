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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations.math_ops import CompareAndBitpack
import mindspore.common.dtype as mstype


class NetCompareAndBitpack(nn.Cell):
    def __init__(self):
        super(NetCompareAndBitpack, self).__init__()
        self.compare_and_bitpack = CompareAndBitpack()

    def construct(self, x, threshold):
        return self.compare_and_bitpack(x, threshold)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_compare_and_bitpack_graph():
    """
    Feature:  Compare and bitpack
    Description: test case for CompareAndBitpack of float16
    Expectation: The result are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float16))
    threshold = Tensor(6, dtype=mstype.float16)
    net = NetCompareAndBitpack()
    output = net(x, threshold)
    out_type = output.asnumpy().dtype
    out_expect = np.array([3], dtype=np.uint8)
    diff0 = output.asnumpy() - out_expect
    error0 = np.zeros(shape=out_expect.shape)
    assert np.all(diff0 == error0)
    assert output.shape == out_expect.shape
    assert out_type == 'uint8'
