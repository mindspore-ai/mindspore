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
from mindspore import Tensor, Parameter


class IndexAddDynNet(nn.Cell):
    def __init__(self, x, axis):
        super(IndexAddDynNet, self).__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.index_add = ops.IndexAdd(axis)
        self.x = x
        self.axis = axis

    def construct(self, indices, y):
        unique_indices, _ = self.unique(indices)
        y = self.gather(y, unique_indices, self.axis)
        return self.index_add(self.x, unique_indices, y)


def dyn_case():
    axis = 1
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([0, 1, 0], dtype=np.int32)
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.float32)
    expect = np.array([[2, 3, 3], [6, 7, 6], [10, 11, 9]], dtype=np.float32)

    net = IndexAddDynNet(Parameter(Tensor(x), name="x"), axis)
    output = net(Tensor(indices), Tensor(y))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_index_add_dyn_cpu():
    """
    Feature: test IndexAdd dynamic shape on CPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_index_add_dyn_gpu():
    """
    Feature: test IndexAdd dynamic shape on GPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_index_add_dyn_ascend():
    """
    Feature: test IndexAdd dynamic shape on Ascend.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_case()
