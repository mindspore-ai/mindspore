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
import pytest
import numpy as np
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.ops.operations import _sequence_ops as S
from mindspore.common import mutable, dtype
from mindspore.ops.composite import GradOperation
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_slice():
    """
    Feature: test sequence_slice op
    Description: slice operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.seq_slice = S.SequenceSlice()

        def construct(self, seq, start, stop, step):
            return self.seq_slice(seq, start, stop, step)

    def func(seq, start, stop, step):
        return seq[start:stop:step]

    seq = (1, 2, 3, 4, 5, 6)
    start = 1
    stop = 3
    step = 1
    net_ms = Net()
    fact = TupleFactory(net_ms, func, (seq, start, stop, step))
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_slice_grad():
    """
    Feature: test sequence_slice grad
    Description: slice operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.seq_slice = S.SequenceSlice()

        def construct(self, seq, start, stop, step):
            return self.seq_slice(seq, start, stop, step)

    seq = mutable((1, 2, 3, 4, 5, 6), True)
    start = 1
    stop = 3
    step = 1
    dout = (1, 1)
    net_ms = Net()
    grad_func = GradOperation(get_all=True, sens_param=True)(net_ms)
    print("grad out1 = ", grad_func(seq, start, stop, step, dout))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_slice_mutable():
    """
    Feature: test sequence_slice mutable
    Description: slice operation on tuple type
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def construct(self, x, a, b):
            out = x[a:b]
            return out

    x = mutable((0, mutable(2), mutable(-3), mutable(-1),
                 mutable(4), -1, -9, mutable(-5), -10))
    a, b = mutable(-9), mutable(-7)
    net = Net()
    out = net(x, a, b)
    ex = x[a:b]
    assert np.allclose(out, ex)


sequence_data_ = [
    mutable([Tensor([1, 2, 3]), Tensor([4, 5, 6]), Tensor([7, 8, 9]),
             Tensor([7, 8, 9]), Tensor([7, 8, 9])], dynamic_len=True),
    mutable([Tensor([1]), Tensor([2]), Tensor([3]), Tensor([5]), Tensor([79])], dynamic_len=True),
    mutable((1, 2, 3, 5, 6, -45, -12, 4), dynamic_len=True),
    mutable([Tensor(1), Tensor(-1), Tensor(45), Tensor(-9878), Tensor(1121), Tensor(1)], dynamic_len=True),
    mutable((mutable(1), mutable(2), mutable(-10), mutable(1), mutable(1), 45, mutable(1)), dynamic_len=True),
    mutable([Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([[1, 2, 3], [4, 5, 6]]),
             Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([[1, 2, 3], [4, 5, 6]])], dynamic_len=True),
    mutable([Tensor([[1, 2, 3], [4, -5, 6]]), Tensor([[1, 2, 3], [4, -5, 6]]), Tensor([[111, 2, 3], [4, -125, -126]]),
             Tensor([[1, 2, 3], [4, 5, 6]]), Tensor([[1, 2, 3], [4, 5, 6]])], dynamic_len=True)
]
start_stop_step_ = [
    (mutable(-100), mutable(5), mutable(2)),
    (-100, 5, 1),
    (-1, -5, -2),
    (-1, -4, -1),
    (5, mutable(2), -2),
    (8, 1, -2),
    (mutable(3), 5, 1),
]
@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("sequence_data", sequence_data_)
@pytest.mark.parametrize("start_stop_step", start_stop_step_)
def test_seq_slice_mutable_and_tensor(sequence_data, start_stop_step):
    """
    Feature: test sequence_slice mutable
    Description: slice operation on tuple type which will cause type error
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def construct(self, x, a, b, c):
            out = x[a:b:c]
            return out

    x = sequence_data
    a, b, c = start_stop_step
    net = Net()
    ex = x[a:b:c]
    out = net(x, a, b, c)
    if isinstance(out[0], (float, int)):
        np.allclose(out, ex)
    else:
        for out_item, ex_item in zip(out, ex):
            assert np.allclose(out_item.asnumpy(), ex_item.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_seq_slice_neg_step():
    """
    Feature: test sequence_slice negative step
    Description: slice operation when step == -1
    Expectation: the behavior is matched to python style
    """
    class Net(nn.Cell):
        def construct(self, x):
            shp = x.shape
            out = shp[::-1]
            return out

    x = Tensor([[2, 1]], dtype.float32)
    net = Net()
    output = net(x)
    expect = x.shape[::-1]
    assert output == expect
