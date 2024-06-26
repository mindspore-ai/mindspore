# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.coalesce = P.Coalesce()

    def construct(self, x_indices, x_values, x_shape):
        return self.coalesce(x_indices, x_values, x_shape)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_dynamic_float32(context_mode):
    """
    Feature: aicpu ops Coalesce.
    Description: test Coalesce forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    x_indices_dyn = Tensor(shape=[2, None], dtype=mstype.int64)
    x_values_dyn = Tensor(shape=[None], dtype=mstype.float32)
    x_shape_dyn = Tensor(shape=[None], dtype=mstype.int64)
    net = Net()
    net.set_inputs(x_indices_dyn, x_values_dyn, x_shape_dyn)
    x_indices = Tensor([[1, 2, 3, 3, 2], [2, 2, 2, 2, 2]], dtype=mstype.int64)
    x_values = Tensor([1, 2, 3, 4, 5], dtype=mstype.float32)
    x_shape = Tensor([5, 5], dtype=mstype.int64)
    y_indices, y_values, y_shape = net(x_indices, x_values, x_shape)
    expect_indices = np.array([[1, 2, 3], [2, 2, 2]]).astype(np.int64)
    expect_values = np.array([1, 7, 7]).astype(np.float32)
    expect_shape = np.array([5, 5]).astype(np.int64)
    assert np.array_equal(y_indices, expect_indices)
    assert np.array_equal(y_values, expect_values)
    assert np.array_equal(y_shape, expect_shape)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_static_double(context_mode):
    """
    Feature: aicpu ops Coalesce.
    Description: test Coalesce forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    x_indices = Tensor([[1, 3, 3], [2, 3, 3]], dtype=mstype.int64)
    x_values = Tensor([1, 1, 1], dtype=mstype.float64)
    x_shape = Tensor([5, 5], dtype=mstype.int64)
    coalesce = Net()
    y_indices, y_values, y_shape = coalesce(x_indices, x_values, x_shape)
    expect_indices = np.array([[1, 3], [2, 3]]).astype(np.int64)
    expect_values = np.array([1, 2]).astype(np.float64)
    expect_shape = np.array([5, 5]).astype(np.int64)
    assert np.array_equal(y_indices, expect_indices)
    assert np.array_equal(y_values, expect_values)
    assert np.array_equal(y_shape, expect_shape)
