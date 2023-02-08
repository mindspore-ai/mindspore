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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops.operations import _sequence_ops as seq
import mindspore.ops as ops
from mindspore import context
from mindspore.common import mutable
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tuple_to_tensor = seq.TupleToTensor()
        self.scalar_to_tensor = ops.ScalarToTensor()

    def construct(self, x, y):
        return self.tuple_to_tensor(x, mstype.float32), self.scalar_to_tensor(y, mstype.int64)


def dyn_case():
    x = mutable((1, 2, 3), True)
    y = mutable(3)
    expect_x = np.array([1, 2, 3], dtype=np.float32)
    expect_y = np.array(3, dtype=np.int64)
    net = Net()
    res_x, res_y = net(x, y)
    rtol = 1.e-4
    atol = 1.e-4
    assert np.allclose(res_x.asnumpy(), expect_x, rtol, atol, equal_nan=True)
    assert np.allclose(res_y.asnumpy(), expect_y, rtol, atol, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_seq_to_tensor():
    """
    Feature: test xxToTensor.
    Description: inputs is dynamic sequence or scalar.
    Expectation: the result match with numpy result
    """
    dyn_case()
