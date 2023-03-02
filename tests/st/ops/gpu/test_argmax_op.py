# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

import random
from functools import reduce
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F


class NetArgmax(nn.Cell):
    def __init__(self, axis=0):
        super(NetArgmax, self).__init__()
        self.argmax = ops.Argmax(axis, output_type=mstype.int32)

    def construct(self, x):
        return self.argmax(x)


class DynRankNet(nn.Cell):
    def __init__(self, axis=0, out_type=mstype.int32):
        super(DynRankNet, self).__init__()
        self.op = ops.Argmax(axis=axis, output_type=out_type)
        self.reduce_sum = ops.ReduceSum(keep_dims=False)

    def construct(self, x, dyn_reduce_axis):
        x = self.reduce_sum(x, dyn_reduce_axis)
        res = self.op(x)
        return res


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_1d():
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x = Tensor(np.array([1., 20., 5.]).astype(np.float32))
        argmax = NetArgmax(axis=0)
        output = argmax(x)
        expect = np.array([1]).astype(np.float32)
        assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_2d():
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x = np.array([[1., 20., 5.],
                      [67., 8., 9.],
                      [130., 24., 15.],
                      [0.3, -0.4, -15.]]).astype(np.float32)
        tensor_x = Tensor(x)
        argmax_axis_0 = NetArgmax(axis=0)
        output = argmax_axis_0(tensor_x)
        expect = np.array([2, 2, 2]).astype(np.int32)
        assert (output.asnumpy() == expect).all()

        argmax_axis_1 = NetArgmax(axis=1)
        output = argmax_axis_1(tensor_x)
        expect = np.array([1, 0, 0, 0]).astype(np.int32)
        assert (output.asnumpy() == expect).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_high_dims():
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")
        for dim in range(3, 10):
            shape = np.random.randint(1, 10, size=dim)
            x = np.random.randn(reduce(lambda x, y: x * y, shape)).astype(np.float32)
            x = x.reshape(shape)

            rnd_axis = random.randint(-dim + 1, dim - 1)
            argmax = NetArgmax(axis=rnd_axis)
            ms_output = argmax(Tensor(x))
            np_output = np.argmax(x, axis=rnd_axis)
            assert (ms_output.asnumpy() == np_output).all()


class ArgmaxFuncNet(nn.Cell):
    def construct(self, x):
        return F.argmax(x, dim=-1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_functional_argmax(mode):
    """
    Feature: Test argmax functional api.
    Description: Test argmax functional api for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode, device_target="GPU")
    x = Tensor([[1, 20, 5], [67, 8, 9], [130, 24, 15]], mstype.float32)
    net = ArgmaxFuncNet()
    output = net(x)
    expect_output = np.array([1, 0, 0]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_dynamic_shape():
    """
    Feature: test ops.argmax.
    Description: test dynamic shape of ops.argmax.
    Expectation: the result match with expected result.
    """
    x = np.array([[1., 20., 5.],
                  [67., 8., 9.],
                  [130., 24., 15.],
                  [0.3, -0.4, -15.]]).astype(np.float32)
    tensor_x = Tensor(x)
    # test dynamic shape of argmax.
    dy_shape_argmax_axis_0 = NetArgmax(axis=0)
    input_x_dyn = Tensor(shape=[4, None], dtype=mstype.float32)
    dy_shape_argmax_axis_0.set_inputs(input_x_dyn)
    output = dy_shape_argmax_axis_0(tensor_x)
    expect = np.array([2, 2, 2]).astype(np.float32)
    assert (output.asnumpy() == expect).all()
    # test dynamic_rank
    dyn_rank_net = DynRankNet()
    input_x_dyn = Tensor(shape=[4, None, 1], dtype=mstype.float32)
    dyn_reduce_axis = Tensor(shape=[None], dtype=mstype.int64)
    dyn_rank_net.set_inputs(input_x_dyn, dyn_reduce_axis)

    reduce_axis = Tensor(np.array([-1], dtype=np.int64))
    output = dyn_rank_net(Tensor(np.expand_dims(x, -1)), reduce_axis)
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_support_types():
    """
    Feature: test ops.Argmax.
    Description: test ops.Argmax with different types.
    Expectation: the result match with expected result.
    """
    x_fp16 = Tensor([1., 3., 2.], mstype.float16)
    x_fp32 = Tensor([1., 3., 2.], mstype.float32)
    x_fp64 = Tensor([1., 3., 2.], mstype.float64)
    x_int8 = Tensor([1, 3, 2], mstype.int8)
    x_int16 = Tensor([1, 3, 2], mstype.int16)
    x_int32 = Tensor([1, 3, 2], mstype.int32)
    x_int64 = Tensor([1, 3, 2], mstype.int64)
    x_uint8 = Tensor([1, 3, 2], mstype.uint8)
    x_uint16 = Tensor([1, 3, 2], mstype.uint16)
    x_uint32 = Tensor([1, 3, 2], mstype.uint32)
    x_uint64 = Tensor([1, 3, 2], mstype.uint64)

    out1_fp16 = ops.Argmax(axis=0, output_type=mstype.int32)(x_fp16)
    out1_fp32 = ops.Argmax(axis=0, output_type=mstype.int32)(x_fp32)
    out1_fp64 = ops.Argmax(axis=0, output_type=mstype.int32)(x_fp64)
    out1_int8 = ops.Argmax(axis=0, output_type=mstype.int32)(x_int8)
    out1_int16 = ops.Argmax(axis=0, output_type=mstype.int32)(x_int16)
    out1_int32 = ops.Argmax(axis=0, output_type=mstype.int32)(x_int32)
    out1_int64 = ops.Argmax(axis=0, output_type=mstype.int32)(x_int64)
    out1_uint8 = ops.Argmax(axis=0, output_type=mstype.int32)(x_uint8)
    out1_uint16 = ops.Argmax(axis=0, output_type=mstype.int32)(x_uint16)
    out1_uint32 = ops.Argmax(axis=0, output_type=mstype.int32)(x_uint32)
    out1_uint64 = ops.Argmax(axis=0, output_type=mstype.int32)(x_uint64)

    out2_fp16 = ops.Argmax(axis=0, output_type=mstype.int64)(x_fp16)
    out2_fp32 = ops.Argmax(axis=0, output_type=mstype.int64)(x_fp32)
    out2_fp64 = ops.Argmax(axis=0, output_type=mstype.int64)(x_fp64)
    out2_int8 = ops.Argmax(axis=0, output_type=mstype.int64)(x_int8)
    out2_int16 = ops.Argmax(axis=0, output_type=mstype.int64)(x_int16)
    out2_int32 = ops.Argmax(axis=0, output_type=mstype.int64)(x_int32)
    out2_int64 = ops.Argmax(axis=0, output_type=mstype.int64)(x_int64)
    out2_uint8 = ops.Argmax(axis=0, output_type=mstype.int64)(x_uint8)
    out2_uint16 = ops.Argmax(axis=0, output_type=mstype.int64)(x_uint16)
    out2_uint32 = ops.Argmax(axis=0, output_type=mstype.int64)(x_uint32)
    out2_uint64 = ops.Argmax(axis=0, output_type=mstype.int64)(x_uint64)

    assert out1_fp16 == 1 and out1_fp16.dtype == mstype.int32
    assert out1_fp32 == 1 and out1_fp32.dtype == mstype.int32
    assert out1_fp64 == 1 and out1_fp64.dtype == mstype.int32
    assert out1_int8 == 1 and out1_int8.dtype == mstype.int32
    assert out1_int16 == 1 and out1_int16.dtype == mstype.int32
    assert out1_int32 == 1 and out1_int32.dtype == mstype.int32
    assert out1_int64 == 1 and out1_int64.dtype == mstype.int32
    assert out1_uint8 == 1 and out1_uint8.dtype == mstype.int32
    assert out1_uint16 == 1 and out1_uint16.dtype == mstype.int32
    assert out1_uint32 == 1 and out1_uint32.dtype == mstype.int32
    assert out1_uint64 == 1 and out1_uint64.dtype == mstype.int32

    assert out2_fp16 == 1 and out2_fp16.dtype == mstype.int64
    assert out2_fp32 == 1 and out2_fp32.dtype == mstype.int64
    assert out2_fp64 == 1 and out2_fp64.dtype == mstype.int64
    assert out2_int8 == 1 and out2_int8.dtype == mstype.int64
    assert out2_int16 == 1 and out2_int16.dtype == mstype.int64
    assert out2_int32 == 1 and out2_int32.dtype == mstype.int64
    assert out2_int64 == 1 and out2_int64.dtype == mstype.int64
    assert out2_uint8 == 1 and out2_uint8.dtype == mstype.int64
    assert out2_uint16 == 1 and out2_uint16.dtype == mstype.int64
    assert out2_uint32 == 1 and out2_uint32.dtype == mstype.int64
    assert out2_uint64 == 1 and out2_uint64.dtype == mstype.int64


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_argmax_functional():
    """
    Feature: test ops.argmax.
    Description: test ops.argmax functional api.
    Expectation: the result match with expected result.
    """
    x = Tensor([[1, 3, 2], [4, 6, 5], [7, 9, 8]], mstype.int32)
    out_dim_none = F.argmax(x, dim=None, keepdim=False)
    out_dim_0 = F.argmax(x, dim=0, keepdim=False)
    out_dim_1 = F.argmax(x, dim=1, keepdim=False)
    out_dim_none_keepdim = F.argmax(x, dim=None, keepdim=True)
    out_dim_0_keepdim = F.argmax(x, dim=0, keepdim=True)
    out_dim_1_keepdim = F.argmax(x, dim=1, keepdim=True)

    assert out_dim_none.asnumpy() == 7
    assert np.all(out_dim_0.asnumpy() == np.array([2, 2, 2]))
    assert np.all(out_dim_1.asnumpy() == np.array([1, 1, 1]))
    assert out_dim_none_keepdim.asnumpy() == 7
    assert np.all(out_dim_0_keepdim.asnumpy() == np.array([[2, 2, 2]]))
    assert np.all(out_dim_1_keepdim.asnumpy() == np.array([[1], [1], [1]]))
