# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
        self.axis5 = (-1,)
        self.axis6 = ()
        self.reduce_mean = P.ReduceMean(False)
        self.reduce_sum = P.ReduceSum(False)
        self.reduce_max = P.ReduceMax(False)
        self.reduce_min = P.ReduceMin(False)

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
                self.reduce_max(indice, self.axis2),
                self.reduce_max(indice, self.axis5),
                self.reduce_max(indice, self.axis6),
                self.reduce_min(indice, self.axis0),
                self.reduce_min(indice, self.axis1),
                self.reduce_min(indice, self.axis2),
                self.reduce_min(indice, self.axis3),
                self.reduce_min(indice, self.axis4),
                self.reduce_min(indice, self.axis5),
                self.reduce_min(indice, self.axis6))


class NetReduceLogic(nn.Cell):
    def __init__(self):
        super(NetReduceLogic, self).__init__()
        self.axis0 = 0
        self.axis1 = -1
        self.axis2 = (0, 1, 2)
        self.axis3 = ()
        self.reduce_all = P.ReduceAll(False)
        self.reduce_any = P.ReduceAny(False)

    @ms_function
    def construct(self, indice):
        return (self.reduce_all(indice, self.axis0),
                self.reduce_all(indice, self.axis1),
                self.reduce_all(indice, self.axis2),
                self.reduce_all(indice, self.axis3),
                self.reduce_any(indice, self.axis0),
                self.reduce_any(indice, self.axis1),
                self.reduce_any(indice, self.axis2),
                self.reduce_any(indice, self.axis3),)


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
    print(output[9])
    print(output[10])
    print(output[11])
    print(output[12])
    print(output[13])
    print(output[14])
    print(output[15])
    print(output[16])
    print(output[17])
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
    expect_9 = np.array([[0., 0., 1., 0., 0., 0.], [1., 0., 0., 0., 1., 0.]]).astype(np.float32)
    expect_10 = np.array([[0., 1., 1., 2., 0., 0.], [1., 0., 0., 4., 0., 1.], [2., 1., 1., 0., 0., 0.]]).astype(
        np.float32)
    expect_11 = np.array([[0., 0.], [0., 0.], [0., 0.]]).astype(np.float32)
    expect_12 = np.array([0., 0., 0., 0., 0., 0.]).astype(np.float32)
    assert (output[0].asnumpy() == expect_0).all()
    assert (output[1].asnumpy() == expect_1).all()
    assert (output[2].asnumpy() == expect_2).all()
    assert (output[3].asnumpy() == expect_3).all()
    assert (output[4].asnumpy() == expect_4).all()
    assert (output[5].asnumpy() == expect_5).all()
    assert (output[6].asnumpy() == expect_6).all()
    assert (output[7].asnumpy() == expect_7).all()
    assert (output[8].asnumpy() == expect_8).all()
    assert (output[9].asnumpy() == expect_8).all()
    assert (output[10].asnumpy() == 5.0).all()
    assert (output[11].asnumpy() == expect_9).all()
    assert (output[12].asnumpy() == expect_10).all()
    assert (output[13].asnumpy() == expect_11).all()
    assert (output[14].asnumpy() == expect_12).all()
    assert (output[15].asnumpy() == 0.0).all()
    assert (output[16].asnumpy() == expect_11).all()
    assert (output[17].asnumpy() == 0.0).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_reduce_logic():
    reduce_logic = NetReduceLogic()
    indice_bool = Tensor([[[False, True, True, True, False, True],
                           [True, True, True, True, True, False]],
                          [[True, False, True, True, False, True],
                           [True, False, False, True, True, True]],
                          [[True, True, True, False, False, False],
                           [True, True, True, False, True, True]]])
    output = reduce_logic(indice_bool)
    expect_all_1 = np.array([[False, False, True, False, False, False],
                             [True, False, False, False, True, False]])
    expect_all_2 = np.array([[False, False], [False, False], [False, False]])
    expect_all_3 = False
    expect_all_4 = False
    expect_any_1 = np.array([[True, True, True, True, False, True], [True, True, True, True, True, True]])
    expect_any_2 = np.array([[True, True], [True, True], [True, True]])
    expect_any_3 = True
    expect_any_4 = True

    assert (output[0].asnumpy() == expect_all_1).all()
    assert (output[1].asnumpy() == expect_all_2).all()
    assert (output[2].asnumpy() == expect_all_3).all()
    assert (output[3].asnumpy() == expect_all_4).all()
    assert (output[4].asnumpy() == expect_any_1).all()
    assert (output[5].asnumpy() == expect_any_2).all()
    assert (output[6].asnumpy() == expect_any_3).all()
    assert (output[7].asnumpy() == expect_any_4).all()


test_reduce()
test_reduce_logic()
