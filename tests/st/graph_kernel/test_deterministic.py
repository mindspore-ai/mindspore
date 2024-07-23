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

import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor, nn
import mindspore.ops as ops
from tests.st.graph_kernel.gk_utils import AssertGKEnable


class ReduceNet(nn.Cell):
    def __init__(self, axis):
        super(ReduceNet, self).__init__()
        self.axis = axis

    def construct(self, x0, x1):
        y0 = ops.Mul()(x0, x1)
        y1 = ops.Mul()(y0, y0)
        y2 = ops.ReduceSum()(y1, self.axis)
        return y2


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_fuse_reduce_deterministic():
    """
    Feature: O1 deterministic test case
    Description: test dvm deterministic
    Expectation: the result of multiple run should be same
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O1"})
    context.set_context(deterministic="ON")
    axis = (0, 1, 2)
    x0 = np.random.normal(0, 1, (1, 5120, 2560)).astype(np.float32)
    x1 = np.random.normal(0, 1, ()).astype(np.float32)
    expect = np.sum(np.square(x0 * x1), axis=axis)
    with AssertGKEnable(True):
        x0_ms = Tensor(x0)
        x1_ms = Tensor(x1)
        net = ReduceNet(axis)
        base_output = net(x0_ms, x1_ms)
        base_output = base_output.asnumpy()
        assert np.allclose(expect, base_output, 1e-4, 1e-4)
        for i in range(9):
            output = net(x0_ms, x1_ms)
            output = output.asnumpy()
            max_diff = np.max(np.abs(output - base_output))
            if max_diff != 0.0:
                raise ValueError("Run[{}] compare failed, max difference {} != 0.0".format(i, max_diff))
