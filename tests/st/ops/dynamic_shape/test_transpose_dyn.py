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


class TransposeDynNet(nn.Cell):
    def __init__(self, axis=0):
        super(TransposeDynNet, self).__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.transpose = ops.Transpose()
        self.axis = axis

    def construct(self, x, perm, indices):
        unique_indices, _ = self.unique(indices)
        input_x = self.gather(x, unique_indices, self.axis)
        return self.transpose(input_x, perm)


def dyn_case():
    perm = (1, 0, 2)
    x = np.arange(2 * 2 * 4).reshape(2, 2, 4).astype(np.float32)
    indices = np.array([0, 1, 0], dtype=np.int32)
    expect = np.array([[[[0, 1, 2, 3],
                         [8, 9, 10, 11]],
                        [[4, 5, 6, 7],
                         [12, 13, 14, 15]]]]).astype(np.float32)

    net = TransposeDynNet()
    output = net(Tensor(x), perm, Tensor(indices))
    assert (output.asnumpy() == expect).all()


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_transpose_dyn_cpu():
    """
    Feature: test Transpose dynamic shape on CPU.
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
def test_transpose_dyn_gpu():
    """
    Feature: test Transpose dynamic shape on GPU.
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
def test_transpose_dyn_ascend():
    """
    Feature: test Transpose dynamic shape on Ascend.
    Description: inputs is dynamic shape.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    dyn_case()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dyn_case()
