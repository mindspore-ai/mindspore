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
import mindspore.ops as ops
from mindspore import Tensor


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batch_to_space_nd_function():
    """
    Feature: test BatchToSpaceND function interface.
    Description: test interface.
    Expectation: the result match with numpy result
    """
    context.set_context(device_target="Ascend")
    x = Tensor(np.arange(4).reshape((4, 1, 1, 1)).astype(
        np.float32), mindspore.float32)
    y = Tensor([2, 2], mindspore.int32)
    z = Tensor([[0, 0], [0, 0]], mindspore.int32)
    output = ops.batch_to_space_nd(x, y, z)
    expect = np.array([[[[0, 1],
                         [2, 3]]]]).astype(np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expect)


class BatchToSpaceNDDynamicShapeNetMS(nn.Cell):
    def __init__(self, axis=1):
        super().__init__()
        self.unique = ops.Unique()
        self.gather = ops.Gather()
        self.batch_to_space_nd = ops.BatchToSpaceNDV2()
        self.axis = axis

    def construct(self, x, block_shape, crops, indices):
        unique_indices, _ = self.unique(indices)
        x = self.gather(x, unique_indices, self.axis)
        return self.batch_to_space_nd(x, block_shape, crops)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_batch_to_space_nd_dynamic():
    """
    Feature: test BatchToSpaceND dynamic shape.
    Description: the input to BatchToSpaceND is dynamic.
    Expectation: the result match with numpy result
    """
    x = np.arange(4).reshape((4, 1, 1, 1)).astype(np.float32)
    y = Tensor([2, 2], mindspore.int32)
    z = Tensor([[0, 0], [0, 0]], mindspore.int32)

    input_x = Tensor(x, mindspore.float32)
    input_y = Tensor(np.array([0, 0, 0, 0]), mindspore.int32)
    expect = np.array([[[[0, 1],
                         [2, 3]]]]).astype(np.float32)
    dyn_net = BatchToSpaceNDDynamicShapeNetMS()

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    output = dyn_net(input_x, y, z, input_y)
    assert (output.asnumpy() == expect).all()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    output = dyn_net(input_x, y, z, input_y)
    assert (output.asnumpy() == expect).all()
