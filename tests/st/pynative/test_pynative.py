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
import torch
import mindspore
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import context
from mindspore.ops.operations import math_ops as MP


class NetTest(Cell):
    def __init__(self):
        super().__init__()
        self.sinc = MP.Sinc()
        self.reduce_sum = P.ReduceMax(keep_dims=False)
        self.relu = P.ReLU()

    def construct(self, x, indices):
        unique_indices = self.relu(indices)
        x = self.reduce_sum(x, unique_indices)
        return self.sinc(x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_pynative_and_graph_mixed_run():
    """
    Feature: test pynative and graph mixed run
    Description: single op run in pynative, the output to net input which run in graph
    Expectation: run success
    """
    context.set_context(jit_level='O0')
    data_x = np.random.randn(7, 3, 8, 8, 8).astype(np.float32)
    x = Tensor(data_x) + 100
    data_indices = np.unique(np.random.randint(2, 4, size=4).astype(np.int32))
    indices = Tensor(data_indices)
    context.set_context(mode=mindspore.GRAPH_MODE, device_target="Ascend")
    out_ms = NetTest()(x, indices)

    y = torch.tensor(data_x) + 100
    indices_pt = torch.tensor(data_indices)
    unique_indices = list(torch.relu(indices_pt).numpy())
    y_reduce = torch.amax(input=y, dim=unique_indices, keepdims=False)
    out_tf = torch.sinc(y_reduce)
    assert np.allclose(out_ms.asnumpy(), out_tf, 0.0001, 0.0001)
