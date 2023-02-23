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
from mindspore import nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import context
from .test_grad_of_dynamic import TestDynamicGrad


class IndexFillNet(nn.Cell):
    def __init__(self):
        super(IndexFillNet, self).__init__()
        self.index_fill = P.array_ops.IndexFill()

    def construct(self, x, dim, index, value):
        out = self.index_fill(x, dim, index, value)
        return out


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_shape_index_fill():
    """
    Feature: IndexFill Grad DynamicShape.
    Description: Test case of dynamic shape for IndexFill grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dim_type = np.int64
    data_type = np.int32
    dim = Tensor(np.array(1, dtype=dim_type))
    value = Tensor(np.array(-10, dtype=data_type))
    x_np = np.random.random(size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    test_dynamic = TestDynamicGrad(IndexFillNet())
    test_dynamic.test_dynamic_grad_net(
        [Tensor(x_np), dim, Tensor(index_np), value], False)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
def test_dynamic_rank_index_fill():
    """
    Feature: IndexFill Grad DynamicShape.
    Description: Test case of dynamic rank for IndexFill grad operator on CPU, GPU and Ascend.
    Expectation: success.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dim_type = np.int64
    data_type = np.int32
    dim = Tensor(np.array(1, dtype=dim_type))
    value = Tensor(np.array(-10, dtype=data_type))
    x_np = np.random.random(size=(5, 5, 5)).astype(data_type)
    index_np = np.random.randint(low=0, high=5, size=4).astype(np.int32)
    test_dynamic = TestDynamicGrad(IndexFillNet())
    test_dynamic.test_dynamic_grad_net(
        [Tensor(x_np), dim, Tensor(index_np), value], True)
