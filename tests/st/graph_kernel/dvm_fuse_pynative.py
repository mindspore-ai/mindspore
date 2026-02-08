# Copyright 2025 Huawei Technologies Co., Ltd
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

"""
dvm fuse op test cases in pynative mode
"""

import numpy as np
from mindspore import ops
from mindspore import context
from mindspore import Tensor
from tests.st.graph_kernel.gk_utils import gen_flag, gen_input, compare_outputs

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)
repeat_num = 2


def test_elemwise():
    """
    Feature: elemwise
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((4, 576, 224, 16), "float32")
    x1 = gen_input((4, 576, 224, 16), "float32")
    x2 = gen_input((4, 576, 224, 16), "float32")
    for i in range(repeat_num):
        y0 = ops.mul(x0, x1)
        y1 = ops.auto_generate.InplaceAddExt()(x2, ops.mul(y0, 5.0))
        compare_outputs("test_elemwise_{}".format(i), [y0, y1])


def test_elemwise_scalar():
    """
    Feature: elemwise scalar
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    for t in ["int32", "float32"]:
        x0 = gen_input((4, 32), t)
        for i in range(repeat_num):
            y0 = ops.add(x0, 4)
            y1 = ops.mul(2, y0)
            compare_outputs(gen_flag("test_elemwise_scalar", t, i), [y1])


def test_elemwise_reduce():
    """
    Feature: elemwise + reduce
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((147456,), "float32")
    x1 = gen_input((147456,), "float32")
    x2 = gen_input((147456,), "float32")
    x3 = gen_input((147456,), "float32")
    for i in range(repeat_num):
        y0 = ops.div(x0, x1)
        y1 = ops.add(x2, y0)
        y2 = ops.mul(y1, x3)
        y3 = ops.auto_generate.SumExt()(y2, (0,), False)
        compare_outputs("test_elemwise_reduce_{}".format(i), y3, cmp_precision=1e-4)


def test_elemwise_broadcast():
    """
    Feature: elemwise + broadcast
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((1, 1, 1, 1), "float32")
    y0 = ops.mul(ops.Tile()(x0, (4, 576, 224, 1)), 1.93762e-6)
    compare_outputs("test_elemwise_broadcast", y0)


def test_broadcast():
    """
    Feature: broadcast
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((1, 3, 8064), "float32")
    x1 = gen_input((1, 3, 3), "float32")
    x2 = gen_input((1, 3, 3), "float32")
    for i in range(repeat_num):
        y0 = ops.Tile()(x0, (4, 1, 1))
        y1 = ops.Tile()(x1, (4, 1, 1))
        y2 = ops.Tile()(x2, (4, 1, 1))
        compare_outputs("test_broadcast_{}".format(i), [y0, y1, y2])


def test_elemwise_reduce_elemwise():
    """
    Feature: elemwise + reduce + elemwise
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((4, 2), "float32")
    y0 = ops.sqrt(ops.auto_generate.SumExt()(ops.mul(x0, x0), (-1,), True))
    compare_outputs("test_elemwise_reduce_elemwise", y0)


def test_view_case():
    """
    Feature: view op
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((8,), "float32")
    x1 = gen_input((4, 2), "float32")
    for i in range(repeat_num):
        y0 = ops.abs(x0)
        y1 = ops.add(x1, ops.reshape(y0, (4, 2)))
        y2 = ops.mul(y1, y0[1:3])
        compare_outputs("test_view_case_{}".format(i), y2)


def test_inplace():
    """
    Feature: inplace op
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((4, 8, 2, 3), "float32")
    x1 = gen_input((2, 3), "float32")
    for i in range(repeat_num):
        y0 = ops.mul(x0, x0)
        y1 = ops.auto_generate.InplaceAddExt()(y0, x1)
        compare_outputs("test_inplace_{}".format(i), y1)


def test_inplace_storage_offset():
    """
    Feature: inplace op, input has storage offset
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = gen_input((16,), "float32")
    x1 = gen_input((16,), "float32")
    x0 = x0[2:8]
    x1 = x1[4:10]
    for i in range(repeat_num):
        y0 = ops.mul(x0, x0)
        y1 = ops.auto_generate.InplaceAddExt()(x1, y0)
        compare_outputs("test_inplace_storage_offset_{}".format(i), [y1, x1])


def test_shape_zero():
    """
    Feature: shape zero case
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    x0 = Tensor(np.random.normal(0, 1, (0, 2)).astype(np.float32))
    x1 = Tensor(np.random.normal(0, 1, (2, 0)).astype(np.float32))
    x2 = gen_input((2, 4), "float32")
    x3 = gen_input((1, 1), "float32")
    for i in range(repeat_num):
        y0 = x0 * 3
        y1 = ops.add(x1, x1)
        y2 = ops.mul(x2, x3)
        print(y0, y1)
        compare_outputs("test_shape_zero_{}".format(i), y2)
