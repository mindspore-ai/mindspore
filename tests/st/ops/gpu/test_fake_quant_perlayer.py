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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
from mindspore.common.tensor import Tensor
import mindspore.nn as nn
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU', device_id=0)


class Net(nn.Cell):
    def __init__(self,
                 num_bits=8,
                 quant_delay=0,
                 symmetric=False,
                 narrow_range=False,
                 training=True):
        super(Net, self).__init__()
        self.fake_quant = Q.FakeQuantPerLayer(num_bits=num_bits,
                                              quant_delay=quant_delay,
                                              symmetric=symmetric,
                                              narrow_range=narrow_range,
                                              training=training)

    def construct(self, x, minq, maxq):
        return self.fake_quant(x, minq, maxq)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant1():
    # (8, false, 0.0f, 0.0f, Shape({2, 3}),
    # {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    # {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(2, 3).astype(np.float32)
    min_val = np.array([0]).reshape(1).astype(np.float32)
    max_val = np.array([0]).reshape(1).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant2():
    # 8, false, -10.0f, 53.75f, Shape({2, 3}),
    # {-10.1f, -10.0f, -9.9f, -9.75f, 53.75f, 53.8f},
    # {-10.0f, -10.0f, -10.0f, -9.75f, 53.75f, 53.75f});
    x = np.array([-10.1, -10.0, -9.9, -9.75, 53.75, 53.8]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-10.0]).reshape(1).astype(np.float32)
    max_val = np.array([53.75]).reshape(1).astype(np.float32)
    expect = np.array([-10.0, -10.0, -10.0, -9.75, 53.75, 53.75]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant3():
    # WithVarsNoNudging_NarrowRange
    x = np.array([-10.1, -10.0, -9.90, -9.75, 53.5, 53.6]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-10.0]).reshape(1).astype(np.float32)
    max_val = np.array([53.5]).reshape(1).astype(np.float32)
    expect = np.array([-10.0, -10.0, -10.0, -9.75, 53.5, 53.5]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant4():
    # WithVarsNudgedDown_RegularRange
    x = np.array([-0.1, 0.0, 0.1, 0.25, 63.75, 63.8]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.1]).reshape(1).astype(np.float32)
    max_val = np.array([63.65]).reshape(1).astype(np.float32)
    expect = np.array([-0.0, 0.0, 0.0, 0.25, 63.75, 63.75]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant5():
    # WithVarsNudgedDown_NarrowRange
    x = np.array([-0.1, 0.0, 0.1, 0.25, 63.5, 63.6]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.1]).reshape(1).astype(np.float32)
    max_val = np.array([63.4]).reshape(1).astype(np.float32)
    expect = np.array([-0.0, 0.0, 0.0, 0.25, 63.5, 63.5]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant6():
    # WithVarsNudgedUp_RegularRange
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.5, 63.6]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.125]).reshape(1).astype(np.float32)
    max_val = np.array([63.625]).reshape(1).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 0.0, 63.5, 63.5]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant7():
    # WithVarsNudgedUp_NarrowRange
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.25, 63.3]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.125]).reshape(1).astype(np.float32)
    max_val = np.array([63.375]).reshape(1).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 0.0, 63.25, 63.25]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant8():
    # WithVarsNudgedZeroIs255_RegularRange
    x = np.array([-63.80, -63.75, -63.70, -63.5, 0.0, 0.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-63.65]).reshape(1).astype(np.float32)
    max_val = np.array([0.1]).reshape(1).astype(np.float32)
    expect = np.array([-63.75, -63.75, -63.75, -63.5, 0.0, 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant9():
    # WithVarsNudgedZeroIs255_NarrowRange
    x = np.array([-63.6, -63.5, -63.4, -63.25, 0.0, 0.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-63.4]).reshape(1).astype(np.float32)
    max_val = np.array([0.1]).reshape(1).astype(np.float32)
    expect = np.array([-63.5, -63.5, -63.5, -63.25, 0.0, 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant10():
    # WithVarsNoNudging_4Bits_RegularRange
    x = np.array([-6.1, -6.0, -5.9, -5.5, 1.5, 1.6]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-6.0]).reshape(1).astype(np.float32)
    max_val = np.array([1.5]).reshape(1).astype(np.float32)
    expect = np.array([-6.0, -6.0, -6.0, -5.5, 1.5, 1.5]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant11():
    # WithVarsNoNudging_4Bits_NarrowRange
    x = np.array([-6.1, -6.0, -5.9, -5.5, 1.0, 1.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-6.0]).reshape(1).astype(np.float32)
    max_val = np.array([1.0]).reshape(1).astype(np.float32)
    expect = np.array([-6.0, -6.0, -6.0, -5.5, 1.0, 1.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant12():
    # WithVarsNudgedDown_4Bits_RegularRange
    x = np.array([-0.1, 0.0, 0.1, 0.5, 7.5, 7.6]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.1]).reshape(1).astype(np.float32)
    max_val = np.array([7.4]).reshape(1).astype(np.float32)
    expect = np.array([-0.0, 0.0, 0.0, 0.5, 7.5, 7.5]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant13():
    # WithVarsNudgedDown_4Bits_NarrowRange
    x = np.array([-0.1, 0.0, 0.1, 0.5, 7.0, 7.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.1]).reshape(1).astype(np.float32)
    max_val = np.array([6.9]).reshape(1).astype(np.float32)
    expect = np.array([-0.0, 0.0, 0.0, 0.5, 7.0, 7.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant14():
    # WithVarsNudgedUp_4Bits_RegularRange
    x = np.array([-0.6, -0.5, -0.24, 0.0, 7.0, 7.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.4]).reshape(1).astype(np.float32)
    max_val = np.array([7.1]).reshape(1).astype(np.float32)
    expect = np.array([-0.5, -0.5, -0.00, 0.0, 7.0, 7.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant15():
    # WithVarsNudgedUp_4Bits_NarrowRange
    x = np.array([-0.6, -0.5, -0.24, 0.0, 6.5, 6.6]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-0.4]).reshape(1).astype(np.float32)
    max_val = np.array([6.6]).reshape(1).astype(np.float32)
    expect = np.array([-0.5, -0.5, -0.00, 0.0, 6.5, 6.5]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant16():
    # WithVarsNudgedZero15_4Bits_RegularRange
    x = np.array([-7.6, -7.5, -7.4, -7.2, 0.0, 0.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-7.3]).reshape(1).astype(np.float32)
    max_val = np.array([0.2]).reshape(1).astype(np.float32)
    expect = np.array([-7.5, -7.5, -7.5, -7.0, 0.0, 0.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant17():
    # WithVarsNudgedZero15_4Bits_NarrowRange
    x = np.array([-7.1, -7.0, -6.9, -6.5, 0.0, 0.1]).reshape(2, 3).astype(np.float32)
    min_val = np.array([-6.8]).reshape(1).astype(np.float32)
    max_val = np.array([0.2]).reshape(1).astype(np.float32)
    expect = np.array([-7.0, -7.0, -7.0, -6.5, 0.0, 0.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)
