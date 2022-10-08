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
from mindspore import ops, nn, context, Tensor
from mindspore.ops.operations import _grad_ops as G
from .test_grad_of_dynamic import TestDynamicGrad


class StridedSliceGradNet(nn.Cell):
    def __init__(self):
        super(StridedSliceGradNet, self).__init__()
        self.op = G.StridedSliceGrad()
        self.shape_op = ops.Shape()
        self.dyn_shape_op = ops.TensorShape()

    def construct(self, dy, x, begin, end, strides):
        x_shape = self.shape_op(x)
        if -1 in x_shape or -2 in x_shape:
            x_shape = self.dyn_shape_op(x)
        return self.op(dy, x_shape, begin, end, strides)


def dyn_grad_func(dtype=np.float16, is_dynamic_rank=False):
    test_dynamic = TestDynamicGrad(StridedSliceGradNet())
    dy = Tensor(np.ones((2, 1, 1)).astype(dtype))
    x = Tensor(
        np.array(
            [
                [[1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4]],
                [[5, 5, 5], [6, 6, 6]],
            ]
        ).astype(dtype)
    )
    begin = (1, 0, 2)
    end = (3, 1, 3)
    strides = (1, 1, 1)
    inputs = [dy, x, begin, end, strides]
    test_dynamic.test_dynamic_grad_net(inputs, is_dynamic_rank=is_dynamic_rank)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_stridedslicegrad_dynamic_shape():
    """
    Feature: Test the bprop process of StridedSliceGrad in PyNative mode with dynamic shape inputs
    Description: The inputs are dynamic shape and the bprop function invokes the operator StridedSlice.
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(dtype=np.float32, is_dynamic_rank=False)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_stridedslicegrad_dynamic_rank():
    """
    Feature: Test the bprop process of StridedSliceGrad in PyNative mode with dynamic rank inputs
    Description: The inputs are dynamic rank and the bprop function invokes the operator StridedSlice.
    Expectation: Assert the result is equal to that of static shape inputs
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    dyn_grad_func(dtype=np.float32, is_dynamic_rank=True)
