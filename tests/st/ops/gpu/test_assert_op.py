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
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, context


context.set_context(mode=context.GRAPH_MODE, device_target='GPU')


class AssertTEST(nn.Cell):
    def __init__(self, summarize):
        super(AssertTEST, self).__init__()
        self.assert1 = P.Assert(summarize)

    def construct(self, cond, x):
        return self.assert1(cond, x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
def test_assert_op():
    """
    Feature: Assert gpu kernel
    Description: test the assert summarize = 10.
    Expectation: match to np benchmark.
    """
    assert1 = AssertTEST(10)
    a = Tensor(np.array([1.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0]).astype(np.float32))
    b = Tensor(np.array([2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0]).astype(np.float16))
    c = Tensor(np.array([3.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0, -1.0, 1.0, 2.0]).astype(np.float64))
    d = Tensor(np.array([4, -4]).astype(np.int16))
    e = Tensor(np.array([5, 6, 7, -4]).astype(np.int32))
    f = Tensor(np.array([5, 6, 7, 5, 6, 7, 5, 6, 7, -4]).astype(np.int64))
    g = Tensor(np.array([6, -4]).astype(np.int8))
    h = Tensor(np.array([7]).astype(np.uint16))
    i = Tensor(np.array([8, 6, 7]).astype(np.uint32))
    j = Tensor(np.array([9, 6, 7, 5, 6, 7, 5, 6, 7]).astype(np.uint64))
    k = Tensor(np.array([10]).astype(np.uint8))
    l = Tensor(np.array([True, False]).astype(np.bool))
    context.set_context(mode=context.GRAPH_MODE)
    assert1(True, [a, b, c, d, e, f, g, h, i, j, k, l])
    context.set_context(mode=context.PYNATIVE_MODE)

    with pytest.raises(RuntimeError) as info:
        assert1(False, [a, b, c, d, e, f, g, h, i, j, k, l])
    assert "assert failed" in str(info.value)
