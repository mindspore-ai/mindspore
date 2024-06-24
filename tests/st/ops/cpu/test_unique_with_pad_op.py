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

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):

    def __init__(self, pad_num):
        super(Net, self).__init__()
        self.uniq = P.UniqueWithPad()
        self.pad_num = pad_num

    def construct(self, x):
        return self.uniq(x, self.pad_num)


def dyn_case():
    net = Net(0)

    x_dyn = Tensor(shape=[None], dtype=mstype.int32)
    net.set_inputs(x_dyn)

    x = Tensor(np.array([1, 1, 2, 2, 3, 3, 4, 5]), dtype=mstype.int32)
    out = net(x)

    expect_shape = (8,)
    for i in range(2):
        assert out[i].asnumpy().shape == expect_shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_dyn():
    """
    Feature: test uniquewithpad in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    dyn_case()
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    dyn_case()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_int32():
    """
    Feature: test uniquewithpad in cpu.
    Description: test uniquewithpad forward with int32 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 5, 2]), mstype.int32)
    uniq = Net(0)
    output = uniq(x)
    expect_y_result = [1, 2, 5, 0]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_float32():
    """
    Feature: test uniquewithpad in cpu.
    Description: test uniquewithpad forward with float32 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 5, 2]), mstype.float32)
    uniq = Net(0.0)
    output = uniq(x)
    expect_y_result = np.array([1, 2, 5, 0]).astype(np.float32)
    expect_idx_result = np.array([0, 1, 2, 1]).astype(np.int32)

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_unique_with_pad_dynamic_shape():
    """
    Feature: uniquewithpad dynamic shape test in cpu.
    Description: test the rightness of uniquewithpad dynamic shape feature.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 5, 2]).astype(np.int32))
    net = Net(0)
    input_dyn = Tensor(shape=[None for _ in x.shape], dtype=x.dtype)
    net.set_inputs(input_dyn)
    output = net(x)
    expect_y_result = [1, 2, 5, 0]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_unique_with_pad_vmap():
    """
    Feature: uniquewithpad vmap test in cpu.
    Description: test the rightness of uniquewithpad vmap feature.
    Expectation: use vmap rule's result equal to manually batched.
    """

    def cal_unique_with_pad(x):
        return P.UniqueWithPad()(x, -1)

    x = Tensor(
        np.array([[[1, 2, 5, 2], [1, 2, 5, 2]],
                  [[1, 2, 5, 2], [1, 2, 5, 2]]]).astype(np.int32))

    vmap_unique_with_pad = vmap(vmap(cal_unique_with_pad, in_axes=0),
                                in_axes=0)
    outputs = vmap_unique_with_pad(x)
    expect0 = np.array([[[1, 2, 5, -1], [1, 2, 5, -1]],
                        [[1, 2, 5, -1], [1, 2, 5, -1]]]).astype(np.int32)
    expect1 = np.array([[[0, 1, 2, 1], [0, 1, 2, 1]],
                        [[0, 1, 2, 1], [0, 1, 2, 1]]]).astype(np.int32)
    assert np.allclose(outputs[0].asnumpy(), expect0)
    assert np.allclose(outputs[1].asnumpy(), expect1)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_int64():
    """
    Feature: test uniquewithpad in cpu.
    Description: test uniquewithpad forward with int64 dtype.
    Expectation: expect correct forward result.
    """
    x = Tensor(np.array([1, 2, 5, 2]), mstype.int64)
    uniq = Net(0)
    output = uniq(x)
    expect_y_result = [1, 2, 5, 0]
    expect_idx_result = [0, 1, 2, 1]

    assert (output[0].asnumpy() == expect_y_result).all()
    assert (output[1].asnumpy() == expect_idx_result).all()
