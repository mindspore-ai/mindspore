# Copyright 2021 Huawei Technologies Co., Ltd
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
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.ops.op_info_register import DataType
from mindspore.ops.operations.custom_ops import Custom, CustomRegOp, custom_op_info_register

outer_product_ascend_info = CustomRegOp() \
    .fusion_type("OPAQUE") \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("Ascend") \
    .get_op_info()

outer_product_gpu_info = CustomRegOp() \
    .fusion_type("OPAQUE") \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("GPU") \
    .get_op_info()


@custom_op_info_register(outer_product_ascend_info, outer_product_gpu_info)
def outer_product(a, b):
    c = output_tensor((a.shape[0], b.shape[1]), 'float32')

    for i0 in range(a.shape[0]):
        for i1 in range(b.shape[1]):
            c[i0, i1] = 0.0
            for i2 in range(a.shape[1]):
                c[i0, i1] = c[i0, i1] + (a[i0, i2] * b[i2, i1])
    return c


class TestHybrid(Cell):
    """Net definition"""

    def __init__(self):
        super(TestHybrid, self).__init__()

        def infer_func(x, y):
            return x

        self.program = Custom(outer_product, out_shape=infer_func, out_dtype=infer_func, func_type="akg")

    def construct(self, x, y):
        return self.program(x, y)


def hybrid_case():
    input_x = np.random.normal(0, 1, [4, 4]).astype(np.float32)
    input_y = np.random.normal(0, 1, [4, 4]).astype(np.float32)

    test = TestHybrid()
    output = test(Tensor(input_x), Tensor(input_y))
    expect = np.matmul(input_x, input_y)
    compare_res = np.allclose(expect, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_hybrid_ascend_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: ascend test case, akg dsl using hybrid grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    hybrid_case()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_hybrid_ascend_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: ascend test case, akg dsl using hybrid grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    hybrid_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hybrid_gpu_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: gpu test case, akg dsl using hybrid grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    hybrid_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_hybrid_gpu_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: gpu test case, akg dsl using hybrid grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    hybrid_case()


v_add_ascend_info = CustomRegOp() \
    .fusion_type("OPAQUE") \
    .input(0, "x", "dynamic") \
    .output(0, "y") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .target("Ascend") \
    .get_op_info()

v_add_gpu_info = CustomRegOp() \
    .fusion_type("OPAQUE") \
    .input(0, "x", "dynamic") \
    .output(0, "y") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .target("GPU") \
    .get_op_info()


@custom_op_info_register(v_add_ascend_info, v_add_gpu_info)
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

    def __init__(self, shape):
        super(TestIRbuilder, self).__init__()
        self.program = Custom(v_add, out_shape=shape, out_dtype=mstype.float16, func_type="akg")

    def construct(self, x, y):
        return self.program([x, y])


def irbuider_case():
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float16)
    input_y = np.random.normal(0, 1, shape).astype(np.float16)

    test = TestIRbuilder(shape)
    output = test(Tensor(input_x), Tensor(input_y))
    compare_res = np.allclose(input_x + input_y, output.asnumpy(), 0.001, 0.001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_irbuider_ascend_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: ascend test case, akg dsl using irbuider grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    irbuider_case()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_irbuider_ascend_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: ascend test case, akg dsl using irbuider grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    irbuider_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_irbuider_gpu_graph_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: gpu test case, akg dsl using irbuider grammar in GRAPH_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    irbuider_case()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_irbuider_gpu_pynative_mode():
    """
    Feature: test case for Custom op with func_type="akg"
    Description: gpu test case, akg dsl using irbuider grammar in PYNATIVE_MODE.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    irbuider_case()
