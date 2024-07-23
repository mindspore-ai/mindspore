# Copyright 2023 Huawei Technologies Co., Ltd
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

import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class NetGetItem(nn.Cell):
    def construct(self, seq, idx):
        return seq[idx]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_getitem():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    seq = mutable((Tensor(1), Tensor(2), Tensor(3), Tensor(4)), True)
    idx = 3
    expect = Tensor(4)
    net = NetGetItem()
    res = net(seq, idx)
    assert np.all(res.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_tensor_getitem1():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    seq = mutable((Tensor([[1, 2], [2, 3]]), Tensor([[2, 3], [3, 4]]), Tensor([[3, 4], [4, 5]])), True)
    idx = 2
    expect = Tensor([[3, 4], [4, 5]])
    net = NetGetItem()
    res = net(seq, idx)
    assert np.all(res.asnumpy() == expect.asnumpy())


@arg_mark(plat_marks=['platform_gpu', 'platform_ascend'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_getitem():
    """
    Feature: test sequence getitem op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    seq = mutable((1, 2, 3, 4, 5, 6), True)
    idx = 3
    expect = 4
    net = NetGetItem()
    res = net(seq, idx)
    assert res == expect


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_getitem_grad():
    """
    Feature: test sequence getitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = NetGetItem()
    seq = mutable((1, 2, 3, 4, 5, 6), True)
    index = 1
    dout = 1
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, index, dout))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_getitem_grad_tensor():
    """
    Feature: test sequence getitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = NetGetItem()
    seq = mutable((Tensor([[1, 2], [2, 3]]), Tensor([[2, 3], [3, 4]]), Tensor([[3, 4], [4, 5]])), True)
    index = 1
    dout = Tensor([[2, 3], [3, 4]])
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, index, dout))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_getitem_grad_scalar():
    """
    Feature: test sequence getitem grad op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    net_ms = NetGetItem()
    seq = mutable((mutable(1), 2, mutable(3), 4, 5, 6))
    index = 1
    dout = 1
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, index, dout))
    index = mutable(1)
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, index, dout))
