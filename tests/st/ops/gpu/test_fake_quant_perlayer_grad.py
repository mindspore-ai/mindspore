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
        self.op = Q.FakeQuantPerLayerGrad(num_bits=num_bits, narrow_range=narrow_range)

    def construct(self, dout, x, minq, maxq):
        return self.op(dout, x, minq, maxq)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad1():
    # WithArgsGradient RegularRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.5, 63.6]).astype(np.float32)
    min_val = np.array([-0.125]).reshape(1).astype(np.float32)
    max_val = np.array([63.625]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad2():
    # WithArgsGradient NarrowRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.25, 63.3]).astype(np.float32)
    min_val = np.array([-0.125]).reshape(1).astype(np.float32)
    max_val = np.array([63.375]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad3():
    # WithArgsGradient_4Bits_RegularRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.6, -0.5, -0.4, 0.0, 7.0, 7.1]).astype(np.float32)
    min_val = np.array([-0.4]).reshape(1).astype(np.float32)
    max_val = np.array([7.1]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad4():
    # WithArgsGradient_4Bits_NarrowRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.6, -0.5, -0.4, 0.0, 6.5, 6.6]).astype(np.float32)
    min_val = np.array([-0.4]).reshape(1).astype(np.float32)
    max_val = np.array([6.6]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad5():
    # FakeQuantWithMinMaxVarsGradient
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).astype(np.float32)
    min_val = np.array([0.0]).reshape(1).astype(np.float32)
    max_val = np.array([0.0]).reshape(1).astype(np.float32)
    expect = dout

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad6():
    # WithVarsGradient_RegularRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.5, 63.6]).astype(np.float32)
    min_val = np.array([-0.125]).reshape(1).astype(np.float32)
    max_val = np.array([63.625]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad7():
    # WithVarsGradient_NarrowRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.26, -0.25, -0.24, 0.0, 63.25, 63.3]).astype(np.float32)
    min_val = np.array([-0.125]).reshape(1).astype(np.float32)
    max_val = np.array([63.375]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=8, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad8():
    # WithVarsGradient_4Bits_RegularRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.6, -0.5, -0.4, 0.0, 7.0, 7.1]).astype(np.float32)
    min_val = np.array([-0.4]).reshape(1).astype(np.float32)
    max_val = np.array([7.1]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=False)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fake_quant_grad9():
    # WithVarsGradient_4Bits_NarrowRange
    dout = np.random.uniform(-1, 1, size=[6]).astype('float32')
    x = np.array([-0.6, -0.5, -0.4, 0.0, 6.5, 6.6]).astype(np.float32)
    min_val = np.array([-0.4]).reshape(1).astype(np.float32)
    max_val = np.array([6.6]).reshape(1).astype(np.float32)
    expect = np.array([0.0, dout[1], dout[2], dout[3], dout[4], 0.0]).astype(np.float32)

    net = Net(num_bits=4, narrow_range=True)
    output = net(Tensor(dout), Tensor(x), Tensor(min_val), Tensor(max_val))

    error = np.ones(shape=expect.shape) * 1.0e-5
    diff = output.asnumpy().flatten() - expect
    print("output: ", output)
    print("expect: ", expect)
    assert np.all(np.abs(diff) < error)
