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

import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import mindspore


class ArgMaxWithValueDynNet(nn.Cell):
    def __init__(self):
        super(ArgMaxWithValueDynNet, self).__init__()
        self.arg_max_with_value = ops.ArgMaxWithValue()

    def construct(self, x):
        index, value = self.arg_max_with_value(x)
        return index, value


def dyn_case():
    x = Tensor(np.array([[0.0, 0.4, 0.6], [0.7, 0.1, 0.2], [0.3, 0.9, 0.8]], dtype=np.float32))
    expect_index = np.array([1, 2, 2]).astype(np.int32)
    expect_value = np.array([0.7, 0.9, 0.8]).astype(np.float32)

    input_dynamic = Tensor(shape=[3, None], dtype=mindspore.float32)
    net = ArgMaxWithValueDynNet()
    net.set_inputs(input_dynamic)
    index, value = net(x)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(index.asnumpy(), expect_index, rtol, atol, equal_nan=True)
    assert np.allclose(value.asnumpy(), expect_value, rtol, atol, equal_nan=True)


class ArgMaxWithValueDynNetWithUnique(nn.Cell):
    def __init__(self, axis=0):
        super(ArgMaxWithValueDynNetWithUnique, self).__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.arg_max_with_value = ops.ArgMaxWithValue()
        self.axis = axis

    def construct(self, x, indices):
        unique_indices, _ = self.unique(indices)
        input_x = self.gather(x, unique_indices, self.axis)
        index, value = self.arg_max_with_value(input_x)
        return index, value


def dyn_case_with_unique():
    x = Tensor(np.array([[0.0, 0.4, 0.6], [0.7, 0.1, 0.2], [0.3, 0.9, 0.8]], dtype=np.float32))
    indices = Tensor(np.array([0, 1, 2], dtype=np.int32))
    expect_index = np.array([1, 2, 2]).astype(np.int32)
    expect_value = np.array([0.7, 0.9, 0.8]).astype(np.float32)

    net = ArgMaxWithValueDynNetWithUnique()
    index, value = net(x, indices)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(index.asnumpy(), expect_index, rtol, atol, equal_nan=True)
    assert np.allclose(value.asnumpy(), expect_value, rtol, atol, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_argmax_with_value_dyn_ascend():
    """
    Feature: test ArgmaxWithValue dynamic shape on Ascend.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_case()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_argmax_with_value_with_unique_dyn_ascend():
    """
    Feature: test ArgmaxWithValue dynamic shape on Ascend.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_case_with_unique()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_case_with_unique()
