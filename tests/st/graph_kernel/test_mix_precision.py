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
mix precision test case
"""

import os
import shutil
import numpy as np
import mindspore
from mindspore import ops
from mindspore import context
from mindspore.nn import Cell
from tests.st.graph_kernel.gk_utils import gen_input
from tests.mark_utils import arg_mark

ops_map = {"Add": ops.add,
           "Sub": ops.sub,
           "Mul": ops.mul,
           "Div": ops.div,
           "Minimum": ops.minimum,
           "Maximum": ops.maximum,
           "Select": ops.select,
           "Equal": ops.equal,
           "NotEqual": ops.not_equal,
           "Greater": ops.greater,
           "GreaterEqual": ops.greater_equal,
           "Less": ops.less,
           "LessEqual": ops.less_equal}


class Net(Cell):
    def __init__(self, op_name):
        super().__init__()
        self.op = ops_map[op_name]

    def construct(self, *args):
        return self.op(*args)


class NetBinaryConst(Cell):
    def __init__(self, op_name, const):
        super().__init__()
        self.op = ops_map[op_name]
        self.const = const

    def construct(self, x0):
        return self.op(x0, self.const)


def check_fuse_op(ir_path, op_name):
    def check_op(file_path):
        # pylint: disable=unspecified-encoding
        with open(file_path, 'r') as f:
            for line in f:
                idx1 = line.find("= GraphKernel")
                if idx1 == -1:
                    continue
                idx2 = line.find("(", idx1)
                if line[idx1:idx2].find(op_name) != -1:
                    return True
        return False

    ir_path = os.path.realpath(ir_path)
    for root, _, files in os.walk(ir_path):
        for file_name in files:
            if file_name.find("convert_call_to_prim") != -1:
                if check_op(os.path.join(root, file_name)):
                    return True
    return False


def get_output(op_name, net, inputs, jit_level, ir_path="./ir_mix_precision"):
    context.set_context(jit_config={"jit_level": jit_level})
    if jit_level == "O1":
        os.environ["MS_DEV_DUMP_IR_PASSES"] = "convert_call_to_prim"
        context.set_context(save_graphs=True, save_graphs_path=ir_path)
    output = net(*inputs)
    output = output.float().asnumpy() if output.dtype == mindspore.bfloat16 else output.asnumpy()
    if jit_level == "O1":
        os.environ.pop("MS_DEV_DUMP_IR_PASSES")
        context.set_context(save_graphs=False)
        ret = check_fuse_op(ir_path, op_name)
        if os.path.isdir(ir_path):
            shutil.rmtree(ir_path, ignore_errors=True)
        if not ret:
            raise RuntimeError("Op [{}] should be fused in graph kernel, but not be fused now! inputs: {}"
                               .format(op_name, inputs))
    return output


def run_binary(op_name, types, shapes, eps=0.0):
    for shape in shapes:
        for t0 in types:
            for t1 in types:
                x0 = gen_input(shape[0], t0)
                x1 = gen_input(shape[1], t1)
                output = get_output(op_name, Net(op_name), [x0, x1], "O1")
                expect = get_output(op_name, Net(op_name), [x0, x1], "O0")
                try:
                    np.testing.assert_allclose(output, expect, eps, eps)
                except Exception as e:
                    raise RuntimeError("precision error: {} {} {} {} eps={}".format(op_name, shape, t0, t1, eps)) from e


def run_binary_const(op_name, types, eps=0.0):
    for t0 in types:
        for t1 in types:
            x0 = gen_input((2, 4), t0)
            x1 = gen_input((), t1, is_positive=True)
            output = get_output(op_name, NetBinaryConst(op_name, x1), [x0], "O1")
            expect = get_output(op_name, NetBinaryConst(op_name, x1), [x0], "O0")
            if op_name == "Div":
                # dvm div scalar can not match with acl
                if expect.dtype == mindspore.float32:
                    eps = 1e-4
                else:
                    eps = 1e-3
            try:
                np.testing.assert_allclose(output, expect, eps, eps)
            except Exception as e:
                raise RuntimeError("precision error: {} {} {} eps={}".format(op_name, t0, t1, eps)) from e


def run_select(op_name, types, eps=0.0):
    for shape in [[(2, 4), (2, 4), (2, 4)], [(2, 4), (2, 4), ()], [(2, 4), (), (2, 4)], [(), (), ()]]:
        for t0 in types:
            for t1 in types:
                cond = gen_input(shape[0], "bool")
                x0 = gen_input(shape[1], t0)
                x1 = gen_input(shape[2], t1)
                output = get_output(op_name, Net(op_name), [cond, x0, x1], "O1")
                expect = get_output(op_name, Net(op_name), [cond, x0, x1], "O0")
                try:
                    np.testing.assert_allclose(output, expect, eps, eps)
                except Exception as e:
                    raise RuntimeError("precision error: {} {} {} {} eps={}".format(op_name, shape, t0, t1, eps)) from e


def run_fuse():
    class FuseNet(Cell):
        def construct(self, x0, x1, x2, x3, x4):
            y0 = ops.mul(x0, x1)
            y1 = ops.mul(y0, x2)
            y2 = ops.mul(y0, x3)
            y3 = ops.mul(y0, x4)
            y4 = ops.add(y1, y2)
            y5 = ops.add(y3, y4)
            return y5

    i0 = gen_input((2, 8), "float16")
    i1 = gen_input((), "float32")
    i2 = gen_input((1,), "float16")
    i3 = gen_input((2, 8), "float32")
    i4 = gen_input((2, 8), "float32")
    fuse_op = "Mul_Mul_Mul_Cast_Mul_Add_Add"
    output = get_output(fuse_op, FuseNet(), [i0, i1, i2, i3, i4], "O1", ir_path="ir_mix_precision_fuse")
    expect = get_output(fuse_op, FuseNet(), [i0, i1, i2, i3, i4], "O0")
    np.testing.assert_allclose(output, expect, 0, 0)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_mix_precision():
    """
    Feature: test graph kernel mix precision data type
    Description: inputs may have different data type
    Expectation: the result match with the expected result
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    float_types = ["float16", "float32", "bfloat16"]
    float_int_types = ["float16", "float32", "bfloat16", "int32"]
    run_binary("Add", float_int_types, [[(2, 4), (2, 4)]])
    run_binary("Sub", float_int_types, [[(2, 4), ()]])
    run_binary("Mul", float_int_types, [[(2, 4), (2, 4)], [(2, 4), ()], [(), (2, 4)], [(), ()]])
    run_binary("Minimum", float_int_types, [[(2, 4), (2, 4)]])
    run_binary("Maximum", float_int_types, [[(2, 4), ()]])
    run_binary("Equal", float_int_types, [[(2, 4), (2, 4)]])
    run_binary("NotEqual", float_int_types, [[(2, 4), ()]])
    run_binary("Greater", float_int_types, [[(), (2, 4)]])
    run_binary("GreaterEqual", float_int_types, [[(), ()]])
    run_binary("Less", float_int_types, [[(2, 4), (2, 4)]])
    run_binary("LessEqual", float_int_types, [[(2, 4), (2, 4)]])
    run_binary("Div", float_types, [[(2, 4), (2, 4)], [(2, 4), ()], [(), (2, 4)]])
    run_binary_const("Add", float_int_types)
    run_binary_const("Sub", float_int_types)
    run_binary_const("Mul", float_int_types)
    run_binary_const("Minimum", float_int_types)
    run_binary_const("Maximum", float_int_types)
    run_binary_const("Div", float_types)
    run_select("Select", float_types)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_mix_precision_fuse():
    """
    Feature: test graph kernel mix precision data type
    Description: inputs may have different data type
    Expectation: the result match with the expected result
    """
    np.random.seed(1)
    context.set_context(mode=context.GRAPH_MODE)
    run_fuse()
