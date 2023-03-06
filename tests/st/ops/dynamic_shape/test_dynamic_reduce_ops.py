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
from mindspore import Tensor, context
from mindspore.ops import operations as P


class ReduceMaxMinNet(nn.Cell):
    def __init__(self, axis=()):
        super(ReduceMaxMinNet, self).__init__()
        self.reduce_max = P.ReduceMax()
        self.reduce_min = P.ReduceMin()
        self.axis = axis

    def construct(self, x):
        return self.reduce_max(x, self.axis), self.reduce_min(x, self.axis)


class ReduceSumMeanProdNet(nn.Cell):
    def __init__(self, axis=()):
        super(ReduceSumMeanProdNet, self).__init__()
        self._sum = P.ReduceSum()
        self._mean = P.ReduceMean()
        self._prod = P.ReduceProd()
        self.axis = axis

    def construct(self, x):
        return self._sum(x, self.axis), self._mean(x, self.axis), self._prod(x, self.axis)


class ReduceMaxMinAxisNet(nn.Cell):
    def __init__(self):
        super(ReduceMaxMinAxisNet, self).__init__()
        self.reduce_max = P.ReduceMax()
        self.reduce_min = P.ReduceMin()
        self.tensor_shape = P.TensorShape()

    def construct(self, x, y):
        axis = self.tensor_shape(y)[0:1]
        return self.reduce_max(x, axis), self.reduce_min(x, axis)


class ReduceSumMeanProdAxisNet(nn.Cell):
    def __init__(self):
        super(ReduceSumMeanProdAxisNet, self).__init__()
        self._sum = P.ReduceSum()
        self._mean = P.ReduceMean()
        self._prod = P.ReduceProd()
        self.tensor_shape = P.TensorShape()

    def construct(self, x, y):
        axis = self.tensor_shape(y)[0:1]
        return self._sum(x, axis), self._mean(x, axis), self._prod(x, axis)


def dyn_case(axis, data_type):
    input_np = np.random.uniform(1, 5, (6, 4, 5)).astype(data_type)
    input_ms = Tensor(input_np)
    dynamic_input = Tensor(shape=(None, 4, 5), dtype=input_ms.dtype)

    dy_max_min_net = ReduceMaxMinNet(axis)
    dy_sum_mean_prod_net = ReduceSumMeanProdNet(axis)
    dy_max_min_net.set_inputs(dynamic_input)
    dy_sum_mean_prod_net.set_inputs(dynamic_input)
    max_dyn, min_dyn = dy_max_min_net(input_ms)
    sum_dyn, mean_dyn, prod_dyn = dy_sum_mean_prod_net(input_ms)

    sum_np = np.add.reduce(input_np, axis)
    max_np = np.maximum.reduce(input_np, axis)
    min_np = np.minimum.reduce(input_np, axis)
    mean_np = np.mean(input_np, axis)
    prod_np = np.multiply.reduce(input_np, axis)
    rtol = 1.e-4
    atol = 1.e-4
    np.testing.assert_allclose(sum_dyn.asnumpy(), sum_np, rtol, atol, equal_nan=True)
    np.testing.assert_allclose(max_dyn.asnumpy(), max_np, rtol, atol, equal_nan=True)
    np.testing.assert_allclose(min_dyn.asnumpy(), min_np, rtol, atol, equal_nan=True)
    np.testing.assert_allclose(mean_dyn.asnumpy(), mean_np, 1, 1, equal_nan=True)
    np.testing.assert_allclose(prod_dyn.asnumpy(), prod_np, rtol, atol, equal_nan=True)


def dyn_axis_case(data_type):
    input_np_x = np.random.uniform(1, 5, (6, 4, 5)).astype(data_type)
    input_ms_x = Tensor(input_np_x)
    input_np_y = np.random.uniform(1, 5, (1, 1, 2)).astype(data_type)
    input_ms_y = Tensor(input_np_y)
    dyn_input_x = Tensor(shape=(None, 4, 5), dtype=input_ms_x.dtype)
    dyn_input_y = Tensor(shape=(1, None, 2), dtype=input_ms_y.dtype)

    dy_max_min_net = ReduceMaxMinAxisNet()
    dy_sum_mean_prod_net = ReduceSumMeanProdAxisNet()
    dy_max_min_net.set_inputs(dyn_input_x, dyn_input_y)
    dy_sum_mean_prod_net.set_inputs(dyn_input_x, dyn_input_y)
    max_dyn, min_dyn = dy_max_min_net(input_ms_x, input_ms_y)
    sum_dyn, mean_dyn, prod_dyn = dy_sum_mean_prod_net(input_ms_x, input_ms_y)

    sum_np = np.add.reduce(input_np_x, 1)
    max_np = np.maximum.reduce(input_np_x, 1)
    min_np = np.minimum.reduce(input_np_x, 1)
    mean_np = np.mean(input_np_x, 1)
    prod_np = np.multiply.reduce(input_np_x, 1)
    rtol = 1.e-4
    atol = 1.e-4
    np.testing.assert_allclose(sum_dyn.asnumpy(), sum_np, rtol, atol, equal_nan=True)
    np.testing.assert_allclose(max_dyn.asnumpy(), max_np, rtol, atol, equal_nan=True)
    np.testing.assert_allclose(min_dyn.asnumpy(), min_np, rtol, atol, equal_nan=True)
    np.testing.assert_allclose(mean_dyn.asnumpy(), mean_np, 1, 1, equal_nan=True)
    np.testing.assert_allclose(prod_dyn.asnumpy(), prod_np, rtol, atol, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("data_type", [np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64])
def test_dynamic_reduce(axis, data_type):
    """
    Feature: Reduce DynamicShape.
    Description: Test case of dynamic shape for reduce operator.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_case(axis, data_type)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_case(axis, data_type)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float32, np.float64, np.int32, np.int64, np.uint32, np.uint64])
def test_dynamic_axis_reduce(data_type):
    """
    Feature: Reduce DynamicShape.
    Description: Test case of dynamic shape for reduce operator.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_axis_case(data_type)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_axis_case(data_type)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float32])
def test_dynamic_axis_reduce_ascend(data_type):
    """
    Feature: Reduce DynamicShape.
    Description: Test case of dynamic shape for reduce operator with dynamic axis in Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_axis_case(data_type)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_axis_case(data_type)
