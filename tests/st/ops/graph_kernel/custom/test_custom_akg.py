# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore.ops import DataType, CustomRegOp, custom_info_register


def outer_product(a, b):
    c = output_tensor(a.shape, a.dtype)

    for i0 in range(a.shape[0]):
        for i1 in range(b.shape[1]):
            c[i0, i1] = 0.0
            for i2 in range(a.shape[1]):
                c[i0, i1] = c[i0, i1] + (a[i0, i2] * b[i2, i1])
    return c


def cube(a):
    c = output_tensor(a.shape, a.dtype)
    b = allocate(a.shape, a.dtype, 'local')

    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            b[i0, i1] = a[i0, i1] * a[i0, i1]
            c[i0, i1] = b[i0, i1] * a[i0, i1]

    return c


def multioutput(a, b):
    c = output_tensor(a.shape, a.dtype)
    d = output_tensor(a.shape, a.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
            d[i0, i1] = a[i0, i1] * b[i0, i1]
    return c, d


def custom_inplace_assign_signle_output(a, b):
    c = allocate(a.shape, a.dtype, 'local')
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
            a[i0, i1] = c[i0, i1] * b[i0, i1]
    return a


def custom_inplace_assign_two_outputs(a, b):
    c = allocate(a.shape, a.dtype, 'local')
    d = output_tensor(b.shape, b.dtype)
    for i0 in range(a.shape[0]):
        for i1 in range(a.shape[1]):
            c[i0, i1] = a[i0, i1] + b[i0, i1]
            a[i0, i1] = c[i0, i1] * b[i0, i1]
    for j0 in range(b.shape[0]):
        for j1 in range(b.shape[1]):
            d[j0, j1] = c[j0, j1]
    return a, d


class TestHybridTwoInputs(Cell):
    """Net definition"""

    def __init__(self, func, out_shape, out_dtype):
        super(TestHybridTwoInputs, self).__init__()

        self.program = ops.Custom(func, out_shape=out_shape, out_dtype=out_dtype, func_type="akg")

    def construct(self, x, y):
        return self.program(x, y)


class TestHybridOneInput(Cell):
    """Net definition"""

    def __init__(self, func, out_shape, out_dtype):
        super(TestHybridOneInput, self).__init__()

        self.program = ops.Custom(func, out_shape=out_shape, out_dtype=out_dtype, func_type="akg")

    def construct(self, x):
        return self.program(x)


class TestHybridTwoOutputs(Cell):
    """Net definition"""

    def __init__(self, func, out_shape, out_dtype):
        super(TestHybridTwoOutputs, self).__init__()

        self.program = ops.Custom(func, out_shape=out_shape, out_dtype=out_dtype, func_type="akg")
        self.add = ops.Add()
        self.mul = ops.Mul()

    def construct(self, x, y):
        res1, res2 = self.program(x, y)
        res3 = self.mul(res1, y)
        return self.add(res2, res3)


class MatMulNN(Cell):
    """Net definition"""

    def __init__(self):
        super(MatMulNN, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        return self.matmul(x, y)


class PowNN(Cell):
    """Net definition"""

    def __init__(self):
        super(PowNN, self).__init__()
        self.pow = ops.Pow()

    def construct(self, x):
        return self.pow(x, 3)


def hybrid_outer_product():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybridTwoInputs(outer_product, lambda x, _: x, lambda x, _: x)
    output = test(Tensor(input_x), Tensor(input_y))
    expect = np.matmul(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def hybrid_outer_product_autodiff():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    sens = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybridTwoInputs(outer_product, lambda x, _: x, lambda x, _: x)
    net = MatMulNN()
    dx, dy = ops.GradOperation(sens_param=True, get_all=True)(test)(Tensor(input_x), Tensor(input_y), Tensor(sens))
    edx, edy = ops.GradOperation(sens_param=True, get_all=True)(net)(Tensor(input_x), Tensor(input_y), Tensor(sens))
    compare_res = np.allclose(edx.asnumpy(), dx.asnumpy(), 0.001, 0.001)
    compare_res &= np.allclose(edy.asnumpy(), dy.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def hybrid_pow_autodiff():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    sens = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybridOneInput(cube, lambda x: x, lambda x: x)
    net = PowNN()
    dx = ops.GradOperation(sens_param=True)(test)(Tensor(input_x), Tensor(sens))
    edx = ops.GradOperation(sens_param=True)(net)(Tensor(input_x), Tensor(sens))
    compare_res = np.allclose(edx.asnumpy(), dx.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def hybrid_multioutput_autodiff():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    sens = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybridTwoOutputs(multioutput, lambda x, _: (x, x), lambda x, _: (x, x))
    dx, dy = ops.GradOperation(sens_param=True, get_all=True)(test)(Tensor(input_x), Tensor(input_y), Tensor(sens))
    edx = input_y * sens * 2.0
    edy = input_x * sens * 2.0 + input_y * sens * 2.0
    compare_res = np.allclose(edx, dx.asnumpy(), 0.001, 0.001)
    compare_res &= np.allclose(edy, dy.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def hybrid_custom_inplace_assign_one_output():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybridTwoInputs(custom_inplace_assign_signle_output, lambda x, _: x, lambda x, _: x)
    output = test(Tensor(input_x), Tensor(input_y))
    expect = input_x * input_y + input_y * input_y
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def hybrid_custom_inplace_assign_two_outputs():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybridTwoOutputs(custom_inplace_assign_two_outputs, lambda x, y: (x, y), lambda x, y: (x, y))
    output = test(Tensor(input_x), Tensor(input_y))
    expect = input_x * (input_y**2) + input_y**3 + input_x + input_y
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


@pytest.mark.level1
@pytest.mark.env_onecard
def test_hybrid_ascend_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: ascend test case, akg dsl using hybrid grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    hybrid_outer_product()
    hybrid_custom_inplace_assign_one_output()


@pytest.mark.level1
@pytest.mark.env_onecard
def test_hybrid_ascend_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: ascend test case, akg dsl using hybrid grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    hybrid_outer_product()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hybrid_gpu_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: gpu test case, akg dsl using hybrid grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    hybrid_outer_product()
    hybrid_outer_product_autodiff()
    hybrid_pow_autodiff()
    hybrid_multioutput_autodiff()
    hybrid_custom_inplace_assign_one_output()
    hybrid_custom_inplace_assign_two_outputs()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hybrid_gpu_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: gpu test case, akg dsl using hybrid grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    hybrid_outer_product()
    hybrid_outer_product_autodiff()
    hybrid_pow_autodiff()
    hybrid_multioutput_autodiff()
    hybrid_custom_inplace_assign_one_output()
    hybrid_custom_inplace_assign_two_outputs()


v_add_ascend_info = CustomRegOp()\
    .input(0, "x", "dynamic")\
    .output(0, "y")\
    .dtype_format(DataType.None_None, DataType.None_None)\
    .target("Ascend")\
    .get_op_info()

v_add_gpu_info = CustomRegOp()\
    .input(0, "x", "dynamic")\
    .output(0, "y")\
    .dtype_format(DataType.F16_None, DataType.F16_None)\
    .target("GPU")\
    .get_op_info()


@custom_info_register(v_add_ascend_info, v_add_gpu_info)
def v_add(inputs, attrs):
    def vadd_func(dst, data_1, data_2):
        ib = tvm.ir_builder.create()
        with ib.for_range_n(data_1.shape, "i") as i:
            ib.store(dst, i, ib.load(data_1, i) + ib.load(data_2, i))
        return ib.get()

    data_1, data_2 = inputs[0], inputs[1]
    return tvm.extern(data_1.shape, [data_1, data_2],
                      lambda ins, outs: vadd_func(outs[0], ins[0], ins[1]),
                      name="v_add", dtype=data_1.dtype)


class TestIRbuilder(Cell):
    """Net definition"""

    def __init__(self):
        super(TestIRbuilder, self).__init__()
        self.program = ops.Custom(v_add, out_shape=lambda x: x[0], out_dtype=lambda x: x[0], func_type="akg")

    def construct(self, x, y):
        return self.program([x, y])


def irbuilder_case():
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float16)
    input_y = np.random.normal(0, 1, shape).astype(np.float16)

    test = TestIRbuilder()
    output = test(Tensor(input_x), Tensor(input_y))
    compare_res = np.allclose(input_x + input_y, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


@pytest.mark.level1
@pytest.mark.env_onecard
def test_irbuilder_ascend_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg" and reg info
    Description: ascend test case, akg dsl using irbuilder grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    irbuilder_case()


@pytest.mark.level1
@pytest.mark.env_onecard
def test_irbuilder_ascend_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg" and reg info
    Description: ascend test case, akg dsl using irbuilder grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    irbuilder_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_irbuilder_gpu_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg" and reg info
    Description: gpu test case, akg dsl using irbuilder grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    irbuilder_case()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_irbuilder_gpu_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg" and reg info
    Description: gpu test case, akg dsl using irbuilder grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    irbuilder_case()
