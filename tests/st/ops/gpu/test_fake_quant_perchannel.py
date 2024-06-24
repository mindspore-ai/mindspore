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
from mindspore import nn
from mindspore.ops.operations import _quant_ops as Q

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self, num_bits=8, symmetric=False, narrow_range=False, channel_axis=1):
        super(Net, self).__init__()
        self.op = Q.FakeQuantPerChannel(num_bits=num_bits,
                                        symmetric=symmetric,
                                        narrow_range=narrow_range,
                                        channel_axis=channel_axis)

    def construct(self, x, minq, maxq):
        return self.op(x, minq, maxq)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel1():
    # WithVarsPerChannel_ZeroMinAndMax
    x = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    min_val = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    max_val = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel2():
    # WithVarsPerChannelDim1NudgedDown_RegularRange
    # scale 1/4, zp 0.4, nudge 0. nudged ranges [0.0, 63.75]
    x = np.array([-0.1, 0.0, 63.75, 63.8]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.65, 63.65, 63.65, 63.65]).astype(np.float32)
    expect = np.array([0.0, 0.0, 63.75, 63.75]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel3():
    # WithVarsPerChannelDim1NudgedDown_NarrowRange
    # scale 1/4, zp 1.4, nudge 1. nudged ranges[0.0, 63.5]
    x = np.array([-0.1, 0.0, 63.5, 63.6]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).astype(np.float32)
    max_val = np.array([63.4, 63.4, 63.4, 63.4]).astype(np.float32)
    expect = np.array([0.0, 0.0, 63.5, 63.5]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel4():
    # WithVarsPerChannelDim1NudgedUp_RegularRange
    # [-0.125, 63.625]
    # scale 1/4, zp: 0.5, nudge 0. nudged range [-0.25, 63.5]
    x = np.array([-0.26, -0.25, -0.24, 63.6]).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 63.5]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.625, 63.625, 63.625, 63.625]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel5():
    # WithVarsPerChannelDim1NudgedUp_NarrowRange
    # scale 1/4, zp: 1.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.26, -0.25, -0.24, 63.3]).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 63.25]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]).astype(np.float32)
    max_val = np.array([63.375, 63.375, 63.375, 63.375]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel6():
    # WithVarsPerChannelDim2NudgedDown_RegularRange
    # scale 1/4, zp: 0.4, nudge 0. nudged range [-0.25, 63.75]
    x = np.array([-0.1, 0.0, 0.1, 0.25, 63.75, 63.80]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([-0.0, 0.0, 0.0, 0.25, 63.75, 63.75]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1]).reshape(3).astype(np.float32)
    max_val = np.array([63.65, 63.65, 63.65]).reshape(3).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel7():
    # WithVarsPerChannelDim2NudgedDown_NarrowRange
    # scale 1/4, zp: 1.4, nudge 1. nudged range [-0.25, 63.5]
    x = np.array([-0.1, 0.0, 0.1, 0.25, 63.5, 63.6]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.25, 63.5, 63.5]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1]).reshape(3).astype(np.float32)
    max_val = np.array([63.4, 63.4, 63.4]).reshape(3).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel8():
    # WithVarsPerChannelDim2NudgedUp_RegularRange
    # scale 1/4, zp: 0.5, nudge 1. nudged range [-0.25, 63.5]
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.5, 63.6]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 0.0, 63.5, 63.5]
                      ).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125]).reshape(3).astype(np.float32)
    max_val = np.array([63.625, 63.625, 63.625]).reshape(3).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel9():
    # WithVarsPerChannelDim2NudgedUp_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.25, 63.3]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array(
        [-0.25, -0.25, -0.25, 0.0, 63.25, 63.25]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125]).reshape(3).astype(np.float32)
    max_val = np.array([63.375, 63.375, 63.375]).reshape(3).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel10():
    # WithVarsPerChannelDim4NudgedDown_RegularRange
    # scale 1/4, zp: 0.4, nudge 0. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 0.1, 0.25, 0.5, 0.75,
                  1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
                  63.0, 63.25, 63.5, 63.7, 63.75, 63.8,
                  63.9, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75,
                       1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
                       63.0, 63.25, 63.5, 63.75, 63.75, 63.75,
                       63.75, 63.75, 63.75, 63.75, 63.75, 63.75]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).reshape(4).astype(np.float32)
    max_val = np.array([63.65, 63.65, 63.65, 63.65]
                       ).reshape(4).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect

    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel11():
    # WithVarsPerChannelDim4NudgedDown_NarrowRange
    # scale 1/4, zp: 1.4, nudge 1. nudged range [0.0, 63.25]
    x = np.array([-0.1, 0.0, 0.1, 0.25, 0.5, 0.75,
                  1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
                  63.0, 63.25, 63.3, 63.4, 63.5, 63.6,
                  63.7, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.75,
                       1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
                       63.0, 63.25, 63.25, 63.5, 63.5, 63.5,
                       63.5, 63.5, 63.5, 63.5, 63.5, 63.5]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).reshape(4).astype(np.float32)
    max_val = np.array([63.4, 63.4, 63.4, 63.4]).reshape(4).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel12():
    # WithVarsPerChannelDim4NudgedUp_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.3, -0.25, -0.2, 0.0, 0.25, 0.5,
                  0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                  63.0, 63.25, 63.4, 63.5, 63.6, 63.7,
                  100.0, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 0.0, 0.25, 0.5,
                       0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                       63.0, 63.25, 63.5, 63.5, 63.5, 63.5,
                       63.5, 63.5, 63.5, 63.5, 63.5, 63.5]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]
                       ).reshape(4).astype(np.float32)
    max_val = np.array([63.625, 63.625, 63.625, 63.625]
                       ).reshape(4).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel13():
    # WithVarsPerChannelDim4NudgedUp_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.3, -0.25, -0.2, 0.0, 0.25, 0.5,
                  0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                  63.0, 63.2, 63.25, 63.3, 63.4, 63.5,
                  100.0, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([-0.25, -0.25, -0.25, 0.0, 0.25, 0.5,
                       0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
                       63.0, 63.25, 63.25, 63.25, 63.25, 63.25,
                       63.25, 63.25, 63.25, 63.25, 63.25, 63.25]).astype(np.float32)
    min_val = np.array([-0.125, -0.125, -0.125, -0.125]
                       ).reshape(4).astype(np.float32)
    max_val = np.array([63.375, 63.375, 63.375, 63.375]
                       ).reshape(4).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel14():
    # WithVarsPerChannelDim1NudgedDown_4Bits_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 7.5, 7.6]).reshape(4).astype(np.float32)
    expect = np.array([0.0, 0.0, 7.5, 7.5]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).reshape(4).astype(np.float32)
    max_val = np.array([7.4, 7.4, 7.4, 7.4]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel15():
    # WithVarsPerChannelDim1NudgedDown_4Bits_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 7.0, 7.1]).reshape(4).astype(np.float32)
    expect = np.array([0.0, 0.0, 7.0, 7.0]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).reshape(4).astype(np.float32)
    max_val = np.array([6.9, 6.9, 6.9, 6.9]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel16():
    # WithVarsPerChannelDim1NudgedUp_4Bits_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.6, -0.5, 7.0, 7.1]).reshape(4).astype(np.float32)
    expect = np.array([-0.5, -0.5, 7.0, 7.0]).astype(np.float32)
    min_val = np.array([-0.4, -0.4, -0.4, -0.4]).reshape(4).astype(np.float32)
    max_val = np.array([7.1, 7.1, 7.1, 7.1]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel17():
    # WithVarsPerChannelDim1NudgedUp_4Bits_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.6, -0.5, 6.5, 6.6]).reshape(4).astype(np.float32)
    expect = np.array([-0.5, -0.5, 6.5, 6.5]).astype(np.float32)
    min_val = np.array([-0.4, -0.4, -0.4, -0.4]).reshape(4).astype(np.float32)
    max_val = np.array([6.6, 6.6, 6.6, 6.6]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True, channel_axis=0)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel18():
    # WithVarsPerChannelDim2NudgedDown_4Bits_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 0.1, 0.5, 7.5, 7.6]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.5, 7.5, 7.5]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1]).reshape(3).astype(np.float32)
    max_val = np.array([7.4, 7.4, 7.4]).reshape(3).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel19():
    # WithVarsPerChannelDim2NudgedDown_4Bits_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 0.1, 0.5, 7.0, 7.1]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.5, 7.0, 7.0]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1]).reshape(3).astype(np.float32)
    max_val = np.array([6.9, 6.9, 6.9]).reshape(3).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel20():
    # WithVarsPerChannelDim2NudgedUp_4Bits_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.51, -0.5, -0.24, 0.0, 7.0, 7.1]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([-0.5, -0.5, 0.0, 0.0, 7.0, 7.0]).astype(np.float32)
    min_val = np.array([-0.4, -0.4, -0.4]).reshape(3).astype(np.float32)
    max_val = np.array([7.1, 7.1, 7.1]).reshape(3).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel21():
    # WithVarsPerChannelDim2NudgedUp_4Bits_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.6, -0.5, -0.24, 0.0, 6.5, 6.6]
                 ).reshape(2, 3).astype(np.float32)
    expect = np.array([-0.5, -0.5, 0.0, 0.0, 6.5, 6.5]).astype(np.float32)
    min_val = np.array([-0.4, -0.4, -0.4]).reshape(3).astype(np.float32)
    max_val = np.array([6.6, 6.6, 6.6]).reshape(3).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel22():
    # WithVarsPerChannelDim4NudgedDown_4Bits_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 0.1, 0.5, 1.0, 1.5,
                  1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                  6.0, 6.5, 7.0, 7.4, 7.5, 7.7,
                  7.8, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.5,
                       1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                       6.0, 6.5, 7.0, 7.5, 7.5, 7.5,
                       7.5, 7.5, 7.5, 7.5, 7.5, 7.5]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).reshape(4).astype(np.float32)
    max_val = np.array([7.4, 7.4, 7.4, 7.4]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel23():
    # WithVarsPerChannelDim4NudgedDown_4Bits_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.1, 0.0, 0.1, 0.5, 1.0, 1.5,
                  1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                  6.0, 6.5, 6.8, 6.9, 7.0, 7.1,
                  7.2, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.5,
                       1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                       6.0, 6.5, 7.0, 7.0, 7.0, 7.0,
                       7.0, 7.0, 7.0, 7.0, 7.0, 7.0]).astype(np.float32)
    min_val = np.array([-0.1, -0.1, -0.1, -0.1]).reshape(4).astype(np.float32)
    max_val = np.array([6.9, 6.9, 6.9, 6.9]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel24():
    # WithVarsPerChannelDim4NudgedUp_4Bits_RegularRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.6, -0.5, -0.4, 0.0, 0.5, 1.0,
                  1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                  6.0, 6.5, 6.9, 7.0, 7.1, 7.7,
                  100.0, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([-0.5, -0.5, -0.5, 0.0, 0.5, 1.0,
                       1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                       6.0, 6.5, 7.0, 7.0, 7.0, 7.0,
                       7.0, 7.0, 7.0, 7.0, 7.0, 7.0]).astype(np.float32)
    min_val = np.array([-0.4, -0.4, -0.4, -0.4]).reshape(4).astype(np.float32)
    max_val = np.array([7.1, 7.1, 7.1, 7.1]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_perchannel25():
    # WithVarsPerChannelDim4NudgedUp_4Bits_NarrowRange
    # scale 1/4, zp: 0.5, nudge 2. nudged range [-0.25, 63.25]
    x = np.array([-0.6, -0.5, -0.4, 0.0, 0.5, 1.0,
                  1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                  5.5, 6.0, 6.4, 6.5, 6.6, 6.7,
                  100.0, 100.0, 100.0, 100.0, 100.0, 1000.0]).reshape((1, 4, 2, 3)).astype(np.float32)
    expect = np.array([-0.5, -0.5, -0.5, 0.0, 0.5, 1.0,
                       1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
                       5.5, 6.0, 6.5, 6.5, 6.5, 6.5,
                       6.5, 6.5, 6.5, 6.5, 6.5, 6.5]).astype(np.float32)
    min_val = np.array([-0.4, -0.4, -0.4, -0.4]).reshape(4).astype(np.float32)
    max_val = np.array([6.6, 6.6, 6.6, 6.6]).reshape(4).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True, channel_axis=1)
    output = net(Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)
