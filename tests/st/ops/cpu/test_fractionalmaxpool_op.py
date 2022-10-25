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

import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
import mindspore.ops.operations.nn_ops as ops
import mindspore.ops.operations._grad_ops as grad_ops


class NetFractionalMaxPool(nn.Cell):
    def __init__(self):
        super(NetFractionalMaxPool, self).__init__()
        self.fractional_max_pool = ops.FractionalMaxPool(pooling_ratio=[1.0, 1.5, 1.5, 1.0])

    def construct(self, x):
        return self.fractional_max_pool(x)


class NetFractionalMaxPoolRealRandom(nn.Cell):
    def __init__(self):
        super(NetFractionalMaxPoolRealRandom, self).__init__()
        self.fractional_max_pool = ops.FractionalMaxPool(pooling_ratio=[1.0, 1.5, 1.5, 1.0], deterministic=True,
                                                         pseudo_random=False, seed=5454, seed2=144)

    def construct(self, x):
        return self.fractional_max_pool(x)


class NetFractionalMaxPoolOverlapPing(nn.Cell):
    def __init__(self):
        super(NetFractionalMaxPoolOverlapPing, self).__init__()
        self.fractional_max_pool = ops.FractionalMaxPool(pooling_ratio=[1.0, 1.5, 1.5, 1.0], overlapping=True)

    def construct(self, x):
        return self.fractional_max_pool(x)


class NetFractionalMaxPoolGrad(nn.Cell):
    def __init__(self):
        super(NetFractionalMaxPoolGrad, self).__init__()
        self.fractional_max_pool_grad = grad_ops.FractionalMaxPoolGrad()

    def construct(self, orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence):
        return self.fractional_max_pool_grad(orig_input, orig_output, out_backprop, row_pooling_sequence,
                                             col_pooling_sequence)


class NetFractionalMaxPoolGradOverlapping(nn.Cell):
    def __init__(self):
        super(NetFractionalMaxPoolGradOverlapping, self).__init__()
        self.fractional_max_pool_grad = grad_ops.FractionalMaxPoolGrad(overlapping=True)

    def construct(self, orig_input, orig_output, out_backprop, row_pooling_sequence, col_pooling_sequence):
        return self.fractional_max_pool_grad(orig_input, orig_output, out_backprop, row_pooling_sequence,
                                             col_pooling_sequence)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_fractionalmaxpool_graph():
    """
    Feature: FractionalMaxPool
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    types = [np.float32, np.float64, np.int32, np.int64]
    for type_i in types:
        x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16]).reshape([1, 4, 4, 1]).astype(type_i))
        net = NetFractionalMaxPool()
        output = net(x)
        output_y = output[0].asnumpy()
        output_row_pooling_sequence = output[1].asnumpy()
        output_col_pooling_sequence = output[2].asnumpy()
        expect_output_y = np.array([[[[6], [8]], [[14], [16]]]]).astype(type_i)
        expect_output_row_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        expect_output_col_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        assert np.allclose(output_y, expect_output_y)
        assert np.allclose(output_row_pooling_sequence, expect_output_row_pooling_sequence)
        assert np.allclose(output_col_pooling_sequence, expect_output_col_pooling_sequence)

        net = NetFractionalMaxPoolRealRandom()
        output = net(x)
        type0 = output[0].asnumpy().dtype
        assert type0 == type_i

        net = NetFractionalMaxPoolOverlapPing()
        output = net(x)
        output_y = output[0].asnumpy()
        output_row_pooling_sequence = output[1].asnumpy()
        output_col_pooling_sequence = output[2].asnumpy()
        expect_output_y = np.array([[[[11], [12]], [[15], [16]]]]).astype(type_i)
        expect_output_row_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        expect_output_col_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        assert np.allclose(output_y, expect_output_y)
        assert np.allclose(output_row_pooling_sequence, expect_output_row_pooling_sequence)
        assert np.allclose(output_col_pooling_sequence, expect_output_col_pooling_sequence)

        netgrad = NetFractionalMaxPoolGrad()
        out_backprop = Tensor(np.ones([1, 2, 2, 1]).astype(type_i))
        output_grad = netgrad(x, output[0], out_backprop, output[1], output[2])
        output_grad_y = output_grad[0].asnumpy()
        expect_output_grad_y = np.array([[[[0], [0], [0], [0]], [[0], [1], [0], [1]],
                                          [[0], [0], [0], [0]], [[0], [1], [0], [1]]]]).astype(type_i)
        assert np.allclose(output_grad_y, expect_output_grad_y)

        netgrad = NetFractionalMaxPoolGradOverlapping()
        out_backprop = Tensor(np.ones([1, 2, 2, 1]).astype(type_i))
        output_grad = netgrad(x, output[0], out_backprop, output[1], output[2])
        output_grad_y = output_grad[0].asnumpy()
        expect_output_grad_y = np.array([[[[0], [0], [0], [0]], [[0], [0], [0], [0]],
                                          [[0], [0], [1], [1]], [[0], [0], [1], [1]]]]).astype(type_i)
        assert np.allclose(output_grad_y, expect_output_grad_y)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_fractionalmaxpool_pynative_dynamic():
    """
    Feature: FractionalMaxPool
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    types = [np.float32, np.float64, np.int32, np.int64]
    for type_i in types:
        x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16]).reshape([1, 4, 4, 1]).astype(type_i))
        # case1
        net = NetFractionalMaxPool()
        dy_shape = [None for _ in x.shape]
        input_dyn = Tensor(shape=dy_shape, dtype=x.dtype)
        net.set_inputs(input_dyn)
        output = net(x)
        output_y = output[0].asnumpy()
        output_row_pooling_sequence = output[1].asnumpy()
        output_col_pooling_sequence = output[2].asnumpy()
        expect_output_y = np.array([[[[6], [8]], [[14], [16]]]]).astype(type_i)
        expect_output_row_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        expect_output_col_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        assert np.allclose(output_y, expect_output_y)
        assert np.allclose(output_row_pooling_sequence, expect_output_row_pooling_sequence)
        assert np.allclose(output_col_pooling_sequence, expect_output_col_pooling_sequence)

        # case2
        net = NetFractionalMaxPoolRealRandom()
        dy_shape = [None for _ in x.shape]
        input_dyn = Tensor(shape=dy_shape, dtype=x.dtype)
        net.set_inputs(input_dyn)
        output = net(x)
        type0 = output[0].asnumpy().dtype
        assert type0 == type_i

        # case3
        net = NetFractionalMaxPoolOverlapPing()
        dy_shape = [None for _ in x.shape]
        input_dyn = Tensor(shape=dy_shape, dtype=x.dtype)
        net.set_inputs(input_dyn)
        output = net(x)
        output_y = output[0].asnumpy()
        output_row_pooling_sequence = output[1].asnumpy()
        output_col_pooling_sequence = output[2].asnumpy()
        expect_output_y = np.array([[[[11], [12]], [[15], [16]]]]).astype(type_i)
        expect_output_row_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        expect_output_col_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        assert np.allclose(output_y, expect_output_y)
        assert np.allclose(output_row_pooling_sequence, expect_output_row_pooling_sequence)
        assert np.allclose(output_col_pooling_sequence, expect_output_col_pooling_sequence)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.env_onecard
def test_fractionalmaxpool_pynative():
    """
    Feature: FractionalMaxPool
    Description: Test of input
    Expectation: The results are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    types = [np.float32, np.float64, np.int32, np.int64]
    for type_i in types:
        x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9,
                             10, 11, 12, 13, 14, 15, 16]).reshape([1, 4, 4, 1]).astype(type_i))
        fractionalmaxpool = ops.FractionalMaxPool(pooling_ratio=[1.0, 1.5, 1.5, 1.0])
        output = fractionalmaxpool(x)
        output_y = output[0].asnumpy()
        output_row_pooling_sequence = output[1].asnumpy()
        output_col_pooling_sequence = output[2].asnumpy()
        expect_output_y = np.array([[[[6], [8]], [[14], [16]]]]).astype(type_i)
        expect_output_row_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        expect_output_col_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        assert np.allclose(output_y, expect_output_y)
        assert np.allclose(output_row_pooling_sequence, expect_output_row_pooling_sequence)
        assert np.allclose(output_col_pooling_sequence, expect_output_col_pooling_sequence)

        fractionalmaxpool = ops.FractionalMaxPool(pooling_ratio=[1.0, 1.5, 1.5, 1.0],
                                                  deterministic=True, pseudo_random=False, seed=5454, seed2=144)
        output = fractionalmaxpool(x)
        type0 = output[0].asnumpy().dtype
        assert type0 == type_i

        fractionalmaxpool = ops.FractionalMaxPool(pooling_ratio=[1.0, 1.5, 1.5, 1.0], overlapping=True)
        output = fractionalmaxpool(x)
        output_y = output[0].asnumpy()
        output_row_pooling_sequence = output[1].asnumpy()
        output_col_pooling_sequence = output[2].asnumpy()
        expect_output_y = np.array([[[[11], [12]], [[15], [16]]]]).astype(type_i)
        expect_output_row_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        expect_output_col_pooling_sequence = np.array([0, 2, 4]).astype(np.int64)
        assert np.allclose(output_y, expect_output_y)
        assert np.allclose(output_row_pooling_sequence, expect_output_row_pooling_sequence)
        assert np.allclose(output_col_pooling_sequence, expect_output_col_pooling_sequence)

        fractionalmaxpoolgrad = grad_ops.FractionalMaxPoolGrad()
        out_backprop = Tensor(np.ones([1, 2, 2, 1]).astype(type_i))
        output_grad = fractionalmaxpoolgrad(x, output[0], out_backprop, output[1], output[2])
        output_grad_y = output_grad[0].asnumpy()
        expect_output_grad_y = np.array([[[[0], [0], [0], [0]], [[0], [1], [0], [1]],
                                          [[0], [0], [0], [0]], [[0], [1], [0], [1]]]]).astype(type_i)
        assert np.allclose(output_grad_y, expect_output_grad_y)

        fractionalmaxpoolgrad = grad_ops.FractionalMaxPoolGrad(overlapping=True)
        out_backprop = Tensor(np.ones([1, 2, 2, 1]).astype(type_i))
        output_grad = fractionalmaxpoolgrad(x, output[0], out_backprop, output[1], output[2])
        output_grad_y = output_grad[0].asnumpy()
        expect_output_grad_y = np.array([[[[0], [0], [0], [0]], [[0], [0], [0], [0]],
                                          [[0], [0], [1], [1]], [[0], [0], [1], [1]]]]).astype(type_i)
        assert np.allclose(output_grad_y, expect_output_grad_y)
