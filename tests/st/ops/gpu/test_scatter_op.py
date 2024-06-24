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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scatter_func_indices_out_of_range():
    """
    Feature: test scatter_func invalid indices.
    Description: indices has invalid value.
    Expectation: catch the raised error.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = Parameter(Tensor(np.zeros((2, 3)).astype(np.float32)), name="x")
    indices = Tensor(np.array([[0, 1], [0, 4]]).astype(np.int32))
    updates = Tensor(np.arange(12).reshape((2, 2, 3)).astype(np.float32))
    # div
    with pytest.raises(RuntimeError):
        _ = P.ScatterDiv()(input_x, indices, updates)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scatter_update_output():
    """
    Feature: test ScatterUpdate output and input_x same value.
    Description: check output and input_x value.
    Expectation: output and input_x have same value
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x = Parameter(Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), mindspore.int64), name="x")
    indices = Tensor(np.array([0, 1]), mindspore.int32)
    updates = Tensor(np.array([[2.0, 1.2, 1.0], [3.0, 1.2, 1.0]]), mindspore.int64)
    output = P.ScatterUpdate()(input_x, indices, updates)
    assert np.allclose(output.asnumpy(), input_x.asnumpy(), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scatter_div_0d():
    """
    Feature: test ScatterDiv 0d input.
    Description: 0d input.
    Expectation: the output match with expect
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    input_x_np = np.random.randn()
    indices_np = np.random.randint(0, 1)
    updates_np = np.random.randn()
    input_x = Parameter(Tensor(input_x_np, mindspore.float32), name="x")
    indices = Tensor(indices_np, mindspore.int32)
    updates = Tensor(updates_np, mindspore.float32)
    output = P.ScatterDiv()(input_x, indices, updates)
    expect = input_x_np / updates_np
    assert np.allclose(expect, output.asnumpy(), 0.0001, 0.0001)
