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

import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class Net(nn.Cell):
    def __init__(self, op, axis):
        super(Net, self).__init__()
        self.axis = axis
        if op == "Cummin":
            self.op = ops.cummin
        elif op == "Cummax":
            self.op = ops.cummax
        else:
            raise ValueError("op value error.")

    def construct(self, x):
        return self.op(x, self.axis)


def cum_minmax_compare(op, x, expected, axis, data_type):
    net = Net(op, axis)
    x = np.array(x).astype(data_type)
    expected = (np.array(expected[0]).astype(data_type), np.array(expected[1]).astype(data_type))

    # Pynative
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    output = net(Tensor(x))
    assert np.allclose(output[0].asnumpy(), expected[0], equal_nan=True)
    assert np.allclose(output[1].asnumpy(), expected[1])

    # Graph
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    output = net(Tensor(x))
    assert np.allclose(output[0].asnumpy(), expected[0], equal_nan=True)
    assert np.allclose(output[1].asnumpy(), expected[1])


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.uint8, np.int8, np.int32, np.float16, np.float32])
def test_cummin_multi_dims(data_type):
    """
    Feature: Op Cummin
    Description: test Cummin operator with multiple dimension.
    Expectation: the result match expectation.
    """
    op = "Cummin"
    axis = 1
    x = [[[9, 10, 0, 0, 2], [5, 4, 1, 9, 3], [5, 0, 3, 7, 5], [10, 4, 5, 4, 9]],
         [[5, 0, 8, 8, 10], [9, 0, 1, 5, 2], [9, 5, 8, 9, 7], [10, 9, 2, 2, 2]],
         [[8, 2, 7, 5, 6], [9, 10, 6, 0, 10], [1, 9, 0, 3, 7], [1, 6, 2, 2, 1]]]
    cummin_output = ([[[9, 10, 0, 0, 2], [5, 4, 0, 0, 2], [5, 0, 0, 0, 2], [5, 0, 0, 0, 2]],
                      [[5, 0, 8, 8, 10], [5, 0, 1, 5, 2], [5, 0, 1, 5, 2], [5, 0, 1, 2, 2]],
                      [[8, 2, 7, 5, 6], [8, 2, 6, 0, 6], [1, 2, 0, 0, 6], [1, 2, 0, 0, 1]]],
                     [[[0, 0, 0, 0, 0], [1, 1, 0, 0, 0], [2, 2, 0, 0, 0], [2, 2, 0, 0, 0]],
                      [[0, 0, 0, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 1, 1], [0, 1, 1, 3, 3]],
                      [[0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [2, 0, 2, 1, 0], [3, 0, 2, 1, 3]]])

    cum_minmax_compare(op, x, cummin_output, axis, data_type)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.uint8, np.uint32, np.int8, np.int32, np.int64, np.float16, np.float32])
def test_cummax_multi_dims(data_type):
    """
    Feature: Op Cummax
    Description: test Cummax operator with multiple dimension.
    Expectation: the result match expectation.
    """
    op = "Cummax"
    axis = 1
    x = [[[6, 11, 4, 9, 15], [1, 2, 14, 13, 15], [15, 10, 6, 13, 6], [9, 4, 11, 10, 11]],
         [[5, 1, 5, 13, 7], [19, 4, 14, 11, 14], [5, 15, 6, 20, 0], [6, 2, 4, 15, 16]],
         [[17, 4, 16, 13, 3], [15, 15, 14, 9, 13], [11, 0, 2, 19, 17], [20, 18, 13, 15, 17]]]
    cummax_output = ([[[6, 11, 4, 9, 15], [6, 11, 14, 13, 15], [15, 11, 14, 13, 15], [15, 11, 14, 13, 15]],
                      [[5, 1, 5, 13, 7], [19, 4, 14, 13, 14], [19, 15, 14, 20, 14], [19, 15, 14, 20, 16]],
                      [[17, 4, 16, 13, 3], [17, 15, 16, 13, 13], [17, 15, 16, 19, 17], [20, 18, 16, 19, 17]]],
                     [[[0, 0, 0, 0, 0], [0, 0, 1, 1, 1], [2, 0, 1, 2, 1], [2, 0, 1, 2, 1]],
                      [[0, 0, 0, 0, 0], [1, 1, 1, 0, 1], [1, 2, 1, 2, 1], [1, 2, 1, 2, 3]],
                      [[0, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 1, 0, 2, 2], [3, 3, 0, 2, 3]]])

    cum_minmax_compare(op, x, cummax_output, axis, data_type)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type", [np.float16, np.float32])
def test_cum_minmax_nan(data_type):
    """
    Feature: Op Cummin/Cummax
    Description: test Cummin/Cummax operator with nan input.
    Expectation: the result match expectation.
    """
    inf = float('inf')
    nan = float('nan')
    axis = 0
    x = [4, inf, 1.5, -inf, 0, nan, 1]
    cummin_output = ([4, 4, 1.5, -inf, -inf, nan, nan], [0, 0, 2, 3, 3, 5, 5])
    cummax_output = ([4, inf, inf, inf, inf, nan, nan], [0, 1, 1, 1, 1, 5, 5])

    cum_minmax_compare("Cummin", x, cummin_output, axis, data_type)
    cum_minmax_compare("Cummax", x, cummax_output, axis, data_type)
