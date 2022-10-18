# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.add = P.AddN()

    @jit
    def construct(self, x, y, z):
        return self.add((x, y, z))


class DynRankNet(nn.Cell):
    def __init__(self):
        super(DynRankNet, self).__init__()
        self.op = P.AddN()
        self.reduce_sum = P.ReduceSum(keep_dims=False)

    def construct(self, x, y, z, dyn_reduce_axis):
        x = self.reduce_sum(x, dyn_reduce_axis)
        y = self.reduce_sum(y, dyn_reduce_axis)
        z = self.reduce_sum(z, dyn_reduce_axis)
        res = self.op((x, y, z))
        return res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float32)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = [[[[0., 3., 6., 9.],
                       [12., 15., 18., 21.],
                       [24., 27., 30., 33.]],
                      [[36., 39., 42., 45.],
                       [48., 51., 54., 57.],
                       [60., 63., 66., 69.]],
                      [[72., 75., 78., 81.],
                       [84., 87., 90., 93.],
                       [96., 99., 102., 105.]]]]

    assert (output.asnumpy() == expect_result).all()

    # Test dynamic shape of addn.
    dyn_add = Net()
    input_x_dyn = Tensor(shape=[1, None, 3, 4], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[1, 3, None, 4], dtype=mstype.float32)
    input_z_dyn = Tensor(shape=[1, 3, 3, None], dtype=mstype.float32)
    dyn_add.set_inputs(input_x_dyn, input_y_dyn, input_z_dyn)
    output = add(Tensor(x), Tensor(y), Tensor(z))
    assert (output.asnumpy() == expect_result).all()

    # test dynamic_rank
    dyn_rank_net = DynRankNet()
    input_x_dyn = Tensor(shape=[1, None, 3, 4, 1], dtype=mstype.float32)
    input_y_dyn = Tensor(shape=[1, 3, None, 4, 1], dtype=mstype.float32)
    input_z_dyn = Tensor(shape=[1, 3, 3, None, 1], dtype=mstype.float32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=mstype.int64)
    dyn_rank_net.set_inputs(input_x_dyn, input_y_dyn, input_z_dyn, dyn_reduce_axis)

    reduce_axis = np.array([-1], dtype=np.int64)
    output = dyn_rank_net(Tensor(np.expand_dims(x, -1)), Tensor(np.expand_dims(y, -1)), Tensor(np.expand_dims(z, -1)),
                          Tensor(reduce_axis))
    assert (output.asnumpy() == expect_result).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_float64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.float64)
    assert (output.asnumpy() == expect_result).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.float64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.float64)
    assert (output.asnumpy() == expect_result).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_int64():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.int64)
    assert (output.asnumpy() == expect_result).all()

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    y = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    z = np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4).astype(np.int64)
    add = Net()
    output = add(Tensor(x), Tensor(y), Tensor(z))
    expect_result = np.array([[[[0., 3., 6., 9.],
                                [12., 15., 18., 21.],
                                [24., 27., 30., 33.]],
                               [[36., 39., 42., 45.],
                                [48., 51., 54., 57.],
                                [60., 63., 66., 69.]],
                               [[72., 75., 78., 81.],
                                [84., 87., 90., 93.],
                                [96., 99., 102., 105.]]]]).astype(np.int64)
    assert (output.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_addn_support_type():
    """
    Feature: test ops.addn.
    Description: test ops.addn with different types.
    Expectation: the result match with expected result.
    """
    out_fp16 = ops.addn([Tensor([1.5, 2.5, 3.5], mstype.float16), Tensor([4.5, 5.5, 6.5], mstype.float16)])
    out_fp32 = ops.addn([Tensor([1.5, 2.5, 3.5], mstype.float32), Tensor([4.5, 5.5, 6.5], mstype.float32)])
    out_fp64 = ops.addn([Tensor([1.5, 2.5, 3.5], mstype.float64), Tensor([4.5, 5.5, 6.5], mstype.float64)])
    out_int8 = ops.addn([Tensor([1, 2, 3], mstype.int8), Tensor([4, 5, 6], mstype.int8)])
    out_int16 = ops.addn([Tensor([1, 2, 3], mstype.int16), Tensor([4, 5, 6], mstype.int16)])
    out_int32 = ops.addn([Tensor([1, 2, 3], mstype.int32), Tensor([4, 5, 6], mstype.int32)])
    out_int64 = ops.addn([Tensor([1, 2, 3], mstype.int64), Tensor([4, 5, 6], mstype.int64)])
    out_uint8 = ops.addn([Tensor([1, 2, 3], mstype.uint8), Tensor([4, 5, 6], mstype.uint8)])
    out_uint16 = ops.addn([Tensor([1, 2, 3], mstype.uint16), Tensor([4, 5, 6], mstype.uint16)])
    out_uint32 = ops.addn([Tensor([1, 2, 3], mstype.uint32), Tensor([4, 5, 6], mstype.uint32)])
    out_uint64 = ops.addn([Tensor([1, 2, 3], mstype.uint64), Tensor([4, 5, 6], mstype.uint64)])
    out_complex64 = ops.addn([Tensor(np.asarray(np.complex(1.5 + 0.4j)), mstype.complex64),
                              Tensor(np.asarray(np.complex(2.5 + 0.4j)), mstype.complex64)])
    out_complex128 = ops.addn([Tensor(np.asarray(np.complex(1.5 + 0.4j)), mstype.complex128),
                               Tensor(np.asarray(np.complex(2.5 + 0.4j)), mstype.complex128)])

    assert np.allclose(out_fp16.asnumpy(), Tensor([6., 8., 10.], mstype.float16).asnumpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(out_fp32.asnumpy(), Tensor([6., 8., 10.], mstype.float32).asnumpy(), rtol=1e-5, atol=1e-5)
    assert np.allclose(out_fp64.asnumpy(), Tensor([6., 8., 10.], mstype.float64).asnumpy(), rtol=1e-5, atol=1e-5)
    assert np.all(out_int8.asnumpy() == Tensor([5, 7, 9], mstype.int8).asnumpy())
    assert np.all(out_int16.asnumpy() == Tensor([5, 7, 9], mstype.int16).asnumpy())
    assert np.all(out_int32.asnumpy() == Tensor([5, 7, 9], mstype.int32).asnumpy())
    assert np.all(out_int64.asnumpy() == Tensor([5, 7, 9], mstype.int64).asnumpy())
    assert np.all(out_uint8.asnumpy() == Tensor([5, 7, 9], mstype.uint8).asnumpy())
    assert np.all(out_uint16.asnumpy() == Tensor([5, 7, 9], mstype.uint16).asnumpy())
    assert np.all(out_uint32.asnumpy() == Tensor([5, 7, 9], mstype.uint32).asnumpy())
    assert np.all(out_uint64.asnumpy() == Tensor([5, 7, 9], mstype.uint64).asnumpy())
    assert np.all(out_complex64.asnumpy() == Tensor(np.asarray(np.complex(4 + 0.8j)), mstype.complex64).asnumpy())
    assert np.all(out_complex128.asnumpy() == Tensor(np.asarray(np.complex(4 + 0.8j)), mstype.complex128).asnumpy())
