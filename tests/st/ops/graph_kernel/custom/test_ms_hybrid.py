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

import pytest
import numpy as np
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore.ops import ms_hybrid


@ms_hybrid
def dtype_and_cast_example(a, b):
    d = allocate(a.shape, "float16")
    c = output_tensor(a.shape, "float16")

    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            d[i0, i1] = float16(1.0)
            c[i0, i1] = d[i0, i1] + float16(a[i0, i1])
            c[i0, i1] = c[i0, i1] * float16(b[i0, i1])
    return c


@ms_hybrid
def allocate_and_math_intrin_example(a, b):
    d = allocate(a.shape, a.dtype)
    c = output_tensor(a.shape, a.dtype)

    for i0 in range(a.shape[0]):
        for i1 in range(b.shape[1]):
            d[i0, i1] = exp(a[i0, i1])
            c[i0, i1] = d[i0, i1] + b[i0, i1]
    return c


@ms_hybrid
def grid_example(a, b):
    c = output_tensor(a.shape, a.dtype)

    for arg in grid(a.shape):
        c[arg] = a[arg] + b[arg[0], arg[1]]
    return c


class TestMsHybridDSL(Cell):
    """Net definition"""

    def __init__(self, func, func_type, out_shape=None, out_dtype=None):
        super(TestMsHybridDSL, self).__init__()

        self.program = ops.Custom(func, out_shape=out_shape, out_dtype=out_dtype, func_type=func_type)

    def construct(self, x, y):
        return self.program(x, y)


def ms_hybrid_cast_with_infer():
    np.random.seed(10)
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float16)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float16)

    test = TestMsHybridDSL(dtype_and_cast_example, "hybrid", lambda x, _: x, lambda x, _: x)
    output = test(Tensor(input_x), Tensor(input_y))
    expect = dtype_and_cast_example(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def ms_hybrid_cast_without_infer():
    np.random.seed(10)
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float16)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float16)

    test = TestMsHybridDSL(dtype_and_cast_example, "hybrid")
    output = test(Tensor(input_x), Tensor(input_y))
    expect = dtype_and_cast_example(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def ms_hybrid_cast_pyfunc():
    np.random.seed(10)
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float16)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float16)

    test = TestMsHybridDSL(dtype_and_cast_example, "pyfunc", lambda x, _: x, lambda x, _: x)
    output = test(Tensor(input_x), Tensor(input_y))
    expect = dtype_and_cast_example(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def ms_hybrid_allocate():
    np.random.seed(10)
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float16)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float16)

    test = TestMsHybridDSL(allocate_and_math_intrin_example, "hybrid", lambda x, _: x, lambda x, _: x)
    output = test(Tensor(input_x), Tensor(input_y))
    expect = allocate_and_math_intrin_example(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def ms_hybrid_grid():
    np.random.seed(10)
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float16)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float16)

    test = TestMsHybridDSL(grid_example, "hybrid", lambda x, _: x, lambda x, _: x)
    output = test(Tensor(input_x), Tensor(input_y))
    expect = grid_example(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


@ pytest.mark.level0
@ pytest.mark.platform_arm_ascend_training
@ pytest.mark.platform_x86_ascend_training
@ pytest.mark.env_onecard
def test_ms_hybrid_ascend_graph_mode():
    """
    Feature: test case for Custom op with func_type="ms_hybrid"
    Description: ascend test case, Python DSL with ms_hybrid decorator in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    ms_hybrid_cast_with_infer()
    ms_hybrid_cast_without_infer()
    ms_hybrid_cast_pyfunc()
    ms_hybrid_allocate()


@ pytest.mark.level0
@ pytest.mark.platform_arm_ascend_training
@ pytest.mark.platform_x86_ascend_training
@ pytest.mark.env_onecard
def test_ms_hybrid_ascend_pynative_mode():
    """
    Feature: test case for Custom op with func_type="ms_hybrid"
    Description: ascend test case, Python DSL with ms_hybrid decorator in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    ms_hybrid_cast_with_infer()
    ms_hybrid_cast_without_infer()
    ms_hybrid_cast_pyfunc()
    ms_hybrid_allocate()


@ pytest.mark.level0
@ pytest.mark.platform_x86_gpu_training
@ pytest.mark.env_onecard
def test_ms_hybrid_gpu_graph_mode():
    """
    Feature: test case for Custom op with func_type="ms_hybrid"
    Description: gpu test case, Python DSL with ms_hybrid decorator in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms_hybrid_cast_with_infer()
    ms_hybrid_cast_without_infer()
    ms_hybrid_cast_pyfunc()
    ms_hybrid_allocate()


@ pytest.mark.level0
@ pytest.mark.platform_x86_gpu_training
@ pytest.mark.env_onecard
def test_ms_hybrid_gpu_pynative_mode():
    """
    Feature: test case for Custom op with func_type="ms_hybrid"
    Description: gpu test case, Python DSL with ms_hybrid decorator in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    ms_hybrid_cast_with_infer()
    ms_hybrid_cast_without_infer()
    ms_hybrid_cast_pyfunc()
    ms_hybrid_allocate()
