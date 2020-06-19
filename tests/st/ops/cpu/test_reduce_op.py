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

import pytest
import numpy as np
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.nn as nn
import mindspore.context as context
from mindspore.common.api import ms_function

context.set_context(device_target="CPU")


class NetReduce(nn.Cell):
    def __init__(self):
        super(NetReduce, self).__init__()
        self.axis0 = 0
        self.axis1 = 1
        self.axis2 = -1
        self.axis3 = (0, 1)
        self.axis4 = (0, 1, 2)
        self.reduce_mean = P.ReduceMean(False)
        self.reduce_sum = P.ReduceSum(False)
        self.reduce_max = P.ReduceMax(False)

    @ms_function
    def construct(self, indice):
        return (self.reduce_mean(indice, self.axis0),
                self.reduce_mean(indice, self.axis1),
                self.reduce_mean(indice, self.axis2),
                self.reduce_mean(indice, self.axis3),
                self.reduce_mean(indice, self.axis4),
                self.reduce_sum(indice, self.axis0),
                self.reduce_sum(indice, self.axis2),
                self.reduce_max(indice, self.axis0),
                self.reduce_max(indice, self.axis2))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reduce():
    reduce = NetReduce()
    indice = Tensor(np.array([
        [[0., 2., 1., 4., 0., 2.], [3., 1., 2., 2., 4., 0.]],
        [[2., 0., 1., 5., 0., 1.], [1., 0., 0., 4., 4., 3.]],
        [[4., 1., 4., 0., 0., 0.], [2., 5., 1., 0., 1., 3.]]
    ]).astype(np.float32))
    output = reduce(indice)
    print(output[0])
    print(output[1])
    print(output[2])
    print(output[3])
    print(output[4])
    print(output[5])
    print(output[6])
    print(output[7])
    print(output[8])
    expect_0 = np.array([[2., 1., 2., 3., 0., 1], [2., 2., 1., 2., 3., 2.]]).astype(np.float32)
    expect_1 = np.array([[1.5, 1.5, 1.5, 3., 2., 1.], [1.5, 0., 0.5, 4.5, 2., 2.], [3., 3., 2.5, 0., 0.5, 1.5]]).astype(
        np.float32)
    expect_2 = np.array([[1.5, 2.], [1.5, 2.], [1.5, 2.]]).astype(np.float32)
    expect_3 = np.array([2, 1.5, 1.5, 2.5, 1.5, 1.5]).astype(np.float32)
    expect_4 = np.array([1.75]).astype(np.float32)
    expect_5 = np.array([[6., 3., 6., 9., 0., 3.], [6., 6., 3., 6., 9., 6.]]).astype(np.float32)
    expect_6 = np.array([[9., 12.], [9., 12.], [9., 12.]]).astype(np.float32)
    expect_7 = np.array([[4., 2., 4., 5., 0., 2.], [3., 5., 2., 4., 4., 3.]]).astype(np.float32)
    expect_8 = np.array([[4., 4.], [5., 4.], [4., 5.]]).astype(np.float32)
    assert (output[0].asnumpy() == expect_0).all()
    assert (output[1].asnumpy() == expect_1).all()
    assert (output[2].asnumpy() == expect_2).all()
    assert (output[3].asnumpy() == expect_3).all()
    assert (output[4].asnumpy() == expect_4).all()
    assert (output[5].asnumpy() == expect_5).all()
    assert (output[6].asnumpy() == expect_6).all()
    assert (output[7].asnumpy() == expect_7).all()
    assert (output[8].asnumpy() == expect_8).all()


test_reduce()
