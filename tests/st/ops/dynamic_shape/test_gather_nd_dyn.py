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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class GatherNdDynamicShapeNetMS(nn.Cell):
    def __init__(self, axis=0):
        super().__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.axis = axis
        self.gather_nd = P.GatherNd()

    def construct(self, x, indices_in, indices):
        unique_indices, _ = self.unique(indices)
        input_x = self.gather(x, unique_indices, self.axis)
        return self.gather_nd(input_x, indices_in)


def dyn_case():
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.float32)
    indices_in = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
    indices = Tensor(np.arange(2).astype(np.int32))
    expect_np = np.array([-0.1, 0.5], dtype=np.float32)
    net = GatherNdDynamicShapeNetMS()
    output = net(input_x, indices_in, indices)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(output.asnumpy(), expect_np, rtol, atol, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gather_nd_dyn_cpu():
    """
    Feature: test GatherNd dynamic shape on CPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    dyn_case()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_gather_nd_dyn_ascend():
    """
    Feature: test GatherNd dynamic shape on Ascend.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_nd_dyn_gpu():
    """
    Feature: test GatherNd dynamic shape on GPU.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dyn_case()
