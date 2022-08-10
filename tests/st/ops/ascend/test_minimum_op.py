# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.context as context
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import Tensor


class TwoTensorsMinimum(Cell):
    def __init__(self):
        super(TwoTensorsMinimum, self).__init__()
        self.min = P.Minimum()

    def construct(self, x, y):
        return self.min(x, y)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_minimum_two_tensors_tensor_dynamic():
    """
    Feature: test minimum on ascend in graph mode
    Description: test dynamic shape
    Expectation: result match numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    net = TwoTensorsMinimum()
    input_x_dyn = Tensor(shape=[None, 10], dtype=ms.int32)
    input_x_dyn = Tensor(shape=[None, 10], dtype=ms.int32)

    net.set_inputs(input_x_dyn, input_x_dyn)

    prop = 100 if np.random.random() > 0.5 else 50
    x = np.random.randn(3, 10).astype(np.int32) * prop
    y = np.random.randn(3, 10).astype(np.int32) * prop
    expect = np.minimum(x, y).astype(np.int32)

    output = net(Tensor(x), Tensor(y))
    assert np.all(output.asnumpy() == expect)
