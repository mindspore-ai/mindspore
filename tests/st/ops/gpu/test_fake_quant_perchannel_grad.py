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
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.context as context
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self, num_bits=8, narrow_range=False):
        super(Net, self).__init__()
        self.op = Q.FakeQuantPerChannelGrad(
            num_bits=num_bits, narrow_range=narrow_range)

    def construct(self, dout, x, minq, maxq):
        return self.op(dout, x, minq, maxq)


@pytest.mark.skip(reason="never run on ci or smoke test")
def test_fake_quant_grad1():
    # WithVarsPerChannelDim1GradientNudgedDown_ZeroMinAndMax
    dout = np.random.uniform(-1, 1, size=[4]).astype('float32')
    x = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    min_val = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    max_val = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    expect = dout

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad2():
    # WithVarsPerChannelDim1GradientNudgedDown_RegularRange
    dout = np.random.uniform(-1, 1, size=[4]).astype('float32')
    x = np.array([-0.1, 0.0, 63.75, 63.8]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.65, 63.65, 63.65, 63.65]).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad3():
    # WithVarsPerChannelDim1GradientNudgedDown_NarrowRange
    dout = np.random.uniform(-1, 1, size=[4]).astype('float32')
    x = np.array([-0.1, 0.0, 63.5, 63.6]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.4, 63.4, 63.4, 63.4]).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad4():
    # WithVarsPerChannelDim1GradientNudgedUp_RegularRange
    dout = np.random.uniform(-1, 1, size=[4]).astype('float32')
    x = np.array([-0.3, -0.25, 63.5, 63.6]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.625, 63.625, 63.625, 63.625]).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad5():
    # WithVarsPerChannelDim1GradientNudgedUp_NarrowRange
    dout = np.random.uniform(-1, 1, size=[4]).astype('float32')
    x = np.array([-0.3, -0.25, 63.25, 63.3]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.375, 63.375, 63.375, 63.375]).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad6():
    # WithVarsPerChannelDim2GradientNudgedDown_RegularRange
    read_dout = np.random.uniform(-1, 1, size=[3, 2]).astype('float32')
    x = np.array([-0.1, 0.0, 0.1, 0.25, 63.75, 63.8]
                 ).reshape(3, 2).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.65, 63.65, 63.65]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], dout[3],
                       dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad7():
    # WithVarsPerChannelDim2GradientNudgedDown_NarrowRange
    read_dout = np.random.uniform(-1, 1, size=[3, 2]).astype('float32')
    x = np.array([-0.1, 0.0, 0.1, 0.25, 63.5, 63.6]
                 ).reshape(3, 2).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.4, 63.4, 63.4]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], dout[3],
                       dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad8():
    # WithVarsPerChannelDim2GradientNudgedUp_RegularRange
    read_dout = np.random.uniform(-1, 1, size=[3, 2]).astype('float32')
    x = np.array([-0.3, -0.25, -0.2, 0.0, 63.5, 63.6]
                 ).reshape(3, 2).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.625, 63.625, 63.625]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], dout[3],
                       dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad9():
    # WithVarsPerChannelDim2GradientNudgedUp_NarrowRange
    read_dout = np.random.uniform(-1, 1, size=[3, 2]).astype('float32')
    x = np.array([-0.3, -0.25, -0.2, 0.0, 63.25, 63.3]
                 ).reshape(3, 2).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.375, 63.375, 63.375]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], dout[3],
                       dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad10():
    # WithVarsPerChannelDim4GradientNudgedDown_RegularRange
    read_dout = np.random.uniform(-1, 1, size=[4, 3, 2, 1]).astype('float32')
    x = np.array([-0.1, 0.0, 63.75, 63.8, -0.1, 0.0,
                  63.75, 63.8, -0.1, 0.0, 63.75, 63.8,
                  -0.1, 0.0, 63.75, 63.8, -0.1, 0.0,
                  63.75, 63.8, -0.1, 0.0, 63.75, 63.8]).reshape((4, 3, 2, 1)).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.65, 63.65, 63.65, 63.65]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], 0.0,
                       0.0, dout[5], dout[6], 0.0,
                       0.0, dout[9], dout[10], 0.0,
                       0.0, dout[13], dout[14], 0.0,
                       0.0, dout[17], dout[18], 0.0,
                       0.0, dout[21], dout[22], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad11():
    # WithVarsPerChannelDim4GradientNudgedDown_NarrowRange
    read_dout = np.random.uniform(-1, 1, size=[4, 3, 2, 1]).astype('float32')
    x = np.array([-0.1, 0.0, 63.5, 63.6, -0.1, 0.0, 63.5, 63.6, -0.1, 0.0, 63.5, 63.6, -0.1, 0.0, 63.5,
                  63.6, -0.1, 0.0, 63.5, 63.6, -0.1, 0.0, 63.5, 63.6]).reshape((4, 3, 2, 1)).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.4, 63.4, 63.4, 63.4]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], 0.0,
                       0.0, dout[5], dout[6], 0.0,
                       0.0, dout[9], dout[10], 0.0,
                       0.0, dout[13], dout[14], 0.0,
                       0.0, dout[17], dout[18], 0.0,
                       0.0, dout[21], dout[22], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad12():
    # WithVarsPerChannelDim4GradientNudgedUp_RegularRange
    read_dout = np.random.uniform(-1, 1, size=[4, 3, 2, 1]).astype('float32')
    x = np.array([-0.3, -0.25, 63.5, 63.6, -0.3, -0.25,
                  63.5, 63.6, -0.3, -0.25, 63.5, 63.6,
                  -0.3, -0.25, 63.5, 63.6, -0.3, -0.25,
                  63.5, 63.6, -0.3, -0.25, 63.5, 63.6]).reshape((4, 3, 2, 1)).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.625, 63.625, 63.625, 63.625]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], 0.0,
                       0.0, dout[5], dout[6], 0.0,
                       0.0, dout[9], dout[10], 0.0,
                       0.0, dout[13], dout[14], 0.0,
                       0.0, dout[17], dout[18], 0.0,
                       0.0, dout[21], dout[22], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad13():
    # WithVarsPerChannelDim4GradientNudgedUp_NarrowRange
    read_dout = np.random.uniform(-1, 1, size=[4, 3, 2, 1]).astype('float32')
    x = np.array([-0.3, -0.25, 63.25, 63.3, -0.3, -0.25,
                  63.25, 63.3, -0.3, -0.25, 63.25, 63.3,
                  -0.3, -0.25, 63.25, 63.3, -0.3, -0.25,
                  63.25, 63.3, -0.3, -0.25, 63.25, 63.3]).reshape((4, 3, 2, 1)).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.375, 63.375, 63.375, 63.375]).astype(np.float32)
    dout = read_dout.flatten()
    expect = np.array([0.0, dout[1], dout[2], 0.0,
                       0.0, dout[5], dout[6], 0.0,
                       0.0, dout[9], dout[10], 0.0,
                       0.0, dout[13], dout[14], 0.0,
                       0.0, dout[17], dout[18], 0.0,
                       0.0, dout[21], dout[22], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(read_dout), Tensor(
        x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("=" * 40)
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)
