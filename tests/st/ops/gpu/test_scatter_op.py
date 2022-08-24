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
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.common.dtype import pytype_to_dtype


class ScatterDiv(Cell):
    def __init__(self, input_shape, input_dtype, use_locking):
        super().__init__()
        self.op = P.ScatterDiv(use_locking)
        self.inputdata = Parameter(initializer(1, input_shape, input_dtype), name="input")

    def construct(self, indices, update):
        self.op(self.inputdata, indices, update)
        return self.inputdata


class ScatterMul(Cell):
    def __init__(self, input_shape, input_dtype, use_locking):
        super().__init__()
        self.op = P.ScatterMul(use_locking)
        self.inputdata = Parameter(initializer(1, input_shape, input_dtype), name="input")

    def construct(self, indices, update):
        self.op(self.inputdata, indices, update)
        return self.inputdata


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_scatter_func_small_float32():
    """
    Feature: ScatterDiv/ScatterMul gpu TEST.
    Description: test case for ScatterDiv/ScatterMul
    Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    input_shape = (2, 3)
    input_dtype = np.float32
    update_np = np.array(
        [
            [[23, 11, 15], [14, 36, 215]],
            [[330, 9, 65], [10, 7, 39]]
        ]
    ).astype(np.float32)
    indices_np = np.array([[0, 1], [0, 1]]).astype(np.int32)

    # div
    indices_me = Tensor(indices_np)
    update_me = Tensor(update_np)
    net = ScatterDiv(input_shape, pytype_to_dtype(input_dtype), use_locking=True)
    out = net(indices_me, update_me)
    expect = np.array([[0.00013175, 0.01010101, 0.00102564], [0.00714286, 0.00396825, 0.00011926]])
    assert np.allclose(out.asnumpy(), expect.astype(np.float32), 0.0001, 0.0001)

    # mul
    indices_me = Tensor(indices_np)
    update_me = Tensor(update_np)
    net = ScatterMul(input_shape, pytype_to_dtype(input_dtype), use_locking=True)
    out = net(indices_me, update_me)
    expect = np.array([[7590.0, 99.0, 975.0], [140.0, 252.0, 8385.0]])
    assert np.allclose(out.asnumpy(), expect.astype(np.float32), 0.0001, 0.0001)
