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
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore
from sequence_help import context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class ShapeNet(Cell):
    def __init__(self):
        super().__init__()
        self.shape = mindspore.ops.shape

    def construct(self, x):
        return self.shape(x)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_shape():
    """
    Feature: test sequence shape op
    Description: setitem operation on tuple type
    Expectation: the behavior is matched to python style
    """
    x = Tensor(np.random.randn(1, 2, 4).astype(np.float32))
    dynx = Tensor(shape=(1, 2, None), dtype=mindspore.float32)
    expect_x = (1, 2, 4)
    net = ShapeNet()
    net.set_inputs(dynx)
    res_x = net(x)
    assert expect_x == res_x
