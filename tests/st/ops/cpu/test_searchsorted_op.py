# Copyright 2021 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class SearchSortedNet(nn.Cell):
    def __init__(self, dtype=mstype.int64, right=False):
        super(SearchSortedNet, self).__init__()
        self.searchsorted = P.SearchSorted(dtype=dtype, right=right)

    def construct(self, sequence, values):
        return self.searchsorted(sequence, values)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_right_out32():
    np.random.seed(1)
    input1 = np.sort(np.array(np.random.randint(10, size=(2, 3, 9)), dtype=np.int32), axis=-1)
    sequence = Tensor(input1, mstype.int32)
    input2 = np.array(np.random.randint(10, size=(2, 3, 1)), dtype=np.int32)
    values = Tensor(input2, mstype.int32)

    net = SearchSortedNet(dtype=mstype.int32, right=True)
    output = net(sequence, values)

    expect = [[[9],
               [3],
               [6]],
              [[5],
               [9],
               [8]]]
    assert output.dtype == mstype.int32
    assert (output.asnumpy() == expect).all()

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_out32():
    np.random.seed(1)
    input1 = np.sort(np.array(np.random.randint(10, size=(2, 3, 9)), dtype=np.int64), axis=-1)
    sequence = Tensor(input1, mstype.int64)
    input2 = np.array(np.random.randint(10, size=(2, 3, 1)), dtype=np.int64)
    values = Tensor(input2, mstype.int64)

    net = SearchSortedNet(dtype=mstype.int32, right=False)
    output = net(sequence, values)

    expect = [[[8],
               [0],
               [3]],
              [[5],
               [8],
               [7]]]
    assert output.dtype == mstype.int32
    assert (output.asnumpy() == expect).all()

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_right_out64():
    np.random.seed(1)
    input1 = np.sort(np.array(np.random.random((2, 5)), dtype=np.float32), axis=-1)
    sequence = Tensor(input1, mstype.float32)
    input2 = np.array(np.random.random((2, 3)), dtype=np.float32)
    values = Tensor(input2, mstype.float32)

    net = SearchSortedNet(dtype=mstype.int64, right=True)
    output = net(sequence, values)

    expect = [[4, 4, 2],
              [5, 0, 5]]
    assert output.dtype == mstype.int64
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_out64():
    np.random.seed(1)
    input1 = np.sort(np.array(np.random.random((5)), dtype=np.float64), axis=-1)
    sequence = Tensor(input1, mstype.float64)
    input2 = np.array(np.random.random((2, 3)), dtype=np.float64)
    values = Tensor(input2, mstype.float64)

    net = SearchSortedNet(dtype=mstype.int64, right=False)
    output = net(sequence, values)

    expect = [[1, 2, 3],
              [3, 4, 4]]
    assert output.dtype == mstype.int64
    assert (output.asnumpy() == expect).all()
