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
from mindspore import context, ops, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.op_info_register import TBERegOp, DataType
from mindspore.ops.operations.custom_ops import Custom, CustomRegOp, custom_op_info_register

square_with_bias_op_info = CustomRegOp() \
    .fusion_type("OPAQUE") \
    .attr("bias", "required", "float") \
    .input(0, "x") \
    .output(0, "y") \
    .dtype_format(DataType.F32_Default, DataType.F32_Default) \
    .dtype_format(DataType.F16_Default, DataType.F16_Default) \
    .target("Ascend") \
    .get_op_info()


@custom_op_info_register(square_with_bias_op_info)
def square_with_bias(input_x, output_y, bias=0.0, kernel_name="square_with_bias"):
    import te.lang.cce
    from te import tvm
    from topi import generic
    from topi.cce import util

    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    shape = util.shape_refine(shape)
    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    with tvm.target.cce():
        res0 = te.lang.cce.vmul(data, data)
        res = te.lang.cce.vadds(res0, bias)
        sch = generic.auto_schedule(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res]}

    te.lang.cce.cce_build_code(sch, config)


square_with_bias_v2_op_info = CustomRegOp() \
    .fusion_type("OPAQUE") \
    .attr("bias", "required", "float") \
    .input(0, "input_x") \
    .output(0, "output1") \
    .output(1, "output2") \
    .dtype_format(DataType.F16_Default, DataType.F16_Default, DataType.F16_Default) \
    .dtype_format(DataType.F32_Default, DataType.F32_Default, DataType.F32_Default) \
    .target("Ascend") \
    .get_op_info()


@custom_op_info_register(square_with_bias_v2_op_info)
def square_with_bias_v2(input_x, output1, output2, bias=0.0, kernel_name="square_with_bias_v2"):
    import te.lang.cce
    from te import tvm

    shape = input_x.get("shape")
    dtype = input_x.get("dtype").lower()

    data = tvm.placeholder(shape, name="data", dtype=dtype.lower())

    res0 = te.lang.cce.vmul(data, data)
    res1 = te.lang.cce.vadds(res0, bias)

    with tvm.target.cce():
        sch = te.lang.cce.auto_schedule([res0, res1])

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": [data, res0, res1]}

    te.lang.cce.cce_build_code(sch, config)


add_n_with_bias_op_info = TBERegOp("CustomAddNWithBias") \
    .fusion_type("ELEMWISE") \
    .attr("bias", "required", "float", "all") \
    .input(0, "x", False, "dynamic", "all") \
    .output(0, "y", False, "required", "all") \
    .op_pattern("broadcast") \
    .dtype_format(DataType.F16_None, DataType.F16_None) \
    .dtype_format(DataType.F32_None, DataType.F32_None) \
    .get_op_info()


def add_n_with_bias(inputs, output, bias, kernel_name="add_n_with_bias"):
    import te.lang.cce
    from te import tvm

    if len(inputs) < 2:
        raise ValueError("inputs num should > 2, but got {}".format(len(inputs)))

    data = []
    for i, d in enumerate(inputs):
        shape = d.get("shape")
        dtype = d.get("dtype").lower()
        data.append(tvm.placeholder(shape, name="input_" + str(i), dtype=dtype.lower()))

    res = data[0]
    for i in range(1, len(data)):
        res = te.lang.cce.vadd(res, data[i])
    res = te.lang.cce.vadds(res, bias)

    with tvm.target.cce():
        sch = te.lang.cce.auto_schedule(res)

    data.append(res)

    config = {"print_ir": False,
              "name": kernel_name,
              "tensor_list": data}

    te.lang.cce.cce_build_code(sch, config)


class Net1(Cell):
    """Net definition"""

    def __init__(self):
        super(Net1, self).__init__()
        # TBE dsl with attr
        self.square_with_bias = Custom(square_with_bias, out_shape=[2, 3], out_dtype=mstype.float32, func_type="tbe")
        # TBE dsl with multiple inputs and attr
        self.add_n_with_bias = Custom(add_n_with_bias, out_shape=[2, 3], out_dtype=mstype.float32, func_type="tbe",
                                      reg_info=add_n_with_bias_op_info)
        # TBE dsl with multiple outputs and attr
        self.square_with_bias_v2 = Custom(square_with_bias_v2, out_shape=([2, 3], [2, 3]),
                                          out_dtype=(mstype.float32, mstype.float32), func_type="tbe")
        self.neg = P.Neg()

    def construct(self, x):
        tmp1 = self.square_with_bias(x, 1.0)
        tmp2 = self.square_with_bias(tmp1, 2.0)
        tmp3 = self.neg(tmp2)
        tmp4 = self.add_n_with_bias([tmp1, tmp3], 1.0)
        res = self.square_with_bias_v2(tmp4, 3.0)
        return res


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_multi_input_multi_output_with_attr():
    """
    Feature: ALL To ALL
    Description: test cases for Custom op with multiple inputs, outputs and attr.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    dtype = np.float32
    x = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]).astype(dtype)
    expect0 = np.array([[9.0, 9.0, 9.0], [441.0, 441.0, 441.0]]).astype(dtype)
    expect1 = np.array([[12.0, 12.0, 12.0], [444.0, 444.0, 444.0]]).astype(dtype)
    expect_np = [expect0, expect1]

    net = Net1()
    output = net(Tensor(x))
    if isinstance(output, tuple):
        output_np = [o.asnumpy() for o in output]
    else:
        output_np = [output.asnumpy()]

    if not isinstance(expect_np, list):
        raise TypeError("expect_np should be of type list, but got {}".format(type(expect_np)))
    if not isinstance(output_np, list):
        raise TypeError("output_np should be of type list, but got {}".format(type(output_np)))
    if len(expect_np) != len(output_np):
        raise ValueError("expect_np length {} not equals to output_np length {}".format(len(expect_np), len(output_np)))
    compare_res = []
    for e, o in zip(expect_np, output_np):
        res = np.allclose(e, o, 0.0001, 0.0001)
        compare_res.append(res)
    if not all(compare_res):
        raise ValueError("Precision error, compare result: {}".format(compare_res))


def bprop(data, axis, out, dout):
    gradient = data * 2
    dx = gradient * dout
    return (dx,)


class Net2(Cell):
    """Net definition"""

    def __init__(self):
        super(Net2, self).__init__()
        self.square_with_bias = Custom(square_with_bias, out_shape=[3], out_dtype=mstype.float32, bprop=bprop,
                                       func_type="tbe")

    def construct(self, x):
        res = self.square_with_bias(x, 1.0)
        return res


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_bprop():
    """
    Feature: ALL To ALL
    Description: test cases for bprop function of Custom op.
    Expectation: the result match with numpy result
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    x = np.array([1.0, 4.0, 9.0]).astype(np.float32)
    sens = np.array([1.0, 1.0, 1.0]).astype(np.float32)
    expect = np.array([2.0, 8.0, 18.0]).astype(np.float32)

    net = Net2()
    dx = ops.GradOperation(sens_param=True)(net)(Tensor(x), Tensor(sens))
    dx_np = dx.asnumpy()

    compare_res = np.allclose(expect, dx_np, 0.0001, 0.0001)
    if not compare_res:
        raise ValueError("Precision error, compare result: {}".format(compare_res))
