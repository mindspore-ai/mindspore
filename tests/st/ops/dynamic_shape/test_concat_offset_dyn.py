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
import mindspore.nn as nn
from mindspore.ops.operations import _grad_ops as G
from mindspore import Tensor, context


class ConcatOffsetNet(nn.Cell):
    def __init__(self, axis):
        super(ConcatOffsetNet, self).__init__()
        self.op = G.ConcatOffset(2, axis)

    def construct(self, x0, x1):
        return self.op((x0, x1))


def run_case(run_mode):
    context.set_context(mode=run_mode)
    x0 = Tensor(np.random.uniform(10, 20, (4, 2, 16)).astype(np.float32))
    x1 = Tensor(np.random.uniform(10, 20, (4, 6, 16)).astype(np.float32))
    expect = np.array([[0, 0, 0], [0, 2, 0]]).astype(np.int64)
    x0_dyn = Tensor(shape=[None, None, 16], dtype=mindspore.float32)
    x1_dyn = Tensor(shape=[None, None, 16], dtype=mindspore.float32)
    net = ConcatOffsetNet(1)
    net.set_inputs(x0_dyn, x1_dyn)
    output = net(x0, x1)
    if run_mode == context.GRAPH_MODE:
        assert np.allclose(expect, output.asnumpy())
    else:
        # In PyNative, set_inputs will be ignored. Static shape for ConcatOffset
        # infer output is not a tensor, get constant value output.
        assert np.allclose(expect, output)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_concat_offset():
    """
    Feature: aicpu ConcatOffset
    Description: test ConcatOffset on Ascend.
    Expectation: output compares success with expect.
    """
    context.set_context(device_target="Ascend")
    run_case(context.GRAPH_MODE)
    run_case(context.PYNATIVE_MODE)
