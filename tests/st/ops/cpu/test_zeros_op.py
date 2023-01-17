# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops import function as F
from mindspore.common import dtype as mstype
from mindspore.common import dtype_to_nptype


class ZerosNetDynTensor(nn.Cell):
    def __init__(self):
        super(ZerosNetDynTensor, self).__init__()
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.x = Tensor(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int32))
        self.indices = Tensor(np.array([0, 1, 2, 6, 2, 1], dtype=np.int32))
        self.axis = 0

    def construct(self, dtype):
        unique_indices, _ = self.unique(self.indices)
        input_x = self.gather(self.x, unique_indices, self.axis)
        return F.zeros(input_x, dtype)


def dyn_shape_tensor_run():
    net = ZerosNetDynTensor()
    out = net(mstype.float32)
    expect = np.zeros((1, 2, 3, 7), dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect)


def zeros_func_run(shape, dtype):
    output = F.zeros(shape, dtype)
    expect = np.zeros(shape, dtype_to_nptype(dtype))
    assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_zeros_dynamic_shape():
    """
    Feature: test graph mode
    Description: compare result with numpy
    Expectation: calculate result same to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    dyn_shape_tensor_run()
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="CPU")
    dyn_shape_tensor_run()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_zeros_func_pynative_mode():
    """
    Feature: test pynative mode
    Description: compare result with numpy
    Expectation: calculate result same to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, save_graphs=False, device_target="CPU")
    zeros_func_run((2, 3), mstype.float32)
    zeros_func_run((2,), mstype.float16)
    zeros_func_run((2, 3, 4, 5), mstype.int32)
    zeros_func_run((1, 64), mstype.int8)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu_training
@pytest.mark.env_onecard
def test_zeros_func_graph_mode():
    """
    Feature: test graph mode
    Description: compare result with numpy
    Expectation: calculate result same to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    zeros_func_run((2, 3), mstype.float32)
    zeros_func_run((2,), mstype.float16)
    zeros_func_run((2, 3, 4, 5), mstype.int32)
    zeros_func_run((1, 64), mstype.int8)
