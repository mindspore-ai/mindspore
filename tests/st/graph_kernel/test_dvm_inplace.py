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
Test inplace op.
"""

import numpy as np
import mindspore
from mindspore import ops
from mindspore import context
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from tests.mark_utils import arg_mark


def _output(outputs):
    if not outputs:
        return []
    if not isinstance(outputs, (list, tuple)):
        res = [outputs]
    else:
        res = list(outputs)
    return [output.float().asnumpy() if output.dtype == mindspore.bfloat16 else output.asnumpy() for output in res]


def get_output(net, inputs, parameter_idx, enable_graph_kernel):
    jit_level = "O1" if enable_graph_kernel else "O0"
    context.set_context(jit_config={"jit_level": jit_level})
    outputs = net(*inputs)
    parameter_outputs = []
    for i in parameter_idx:
        parameter_outputs.append(inputs[i])
    return _output(outputs) + _output(parameter_outputs)


def gen_inputs(inputs_shape, inputs_type, parameter_idx):
    inputs = [np.random.normal(0, 1, shape).astype(np.float32) for shape in inputs_shape]
    inputs_ms = []
    for i, inp in enumerate(inputs):
        if i in parameter_idx:
            inputs_ms.append(Parameter(Tensor(np.array(inp), dtype=inputs_type[i]), name="p{}".format(i)))
        else:
            inputs_ms.append(Tensor(np.array(inp), dtype=inputs_type[i]))
    return inputs_ms


def case1():
    class Net(Cell):
        def construct(self, x0, x1, x2, x3, x4):
            y0 = ops.add(x0, x1)
            y1 = ops.assign(x2, y0)
            y2 = ops.div(x3, y1)
            y3 = ops.mul(y2, x4)
            return y1, y3

    parameter_idx = [2]
    inputs = gen_inputs([(3, 4, 2)] * 5, [mindspore.float32] * 5, parameter_idx)
    expects = get_output(Net(), inputs, parameter_idx, False)
    outputs = get_output(Net(), inputs, parameter_idx, True)
    compare_result = [np.allclose(e, o, 1e-5, 1e-5) for e, o in zip(expects, outputs)]
    assert False not in compare_result


def case2():
    class Net(Cell):
        def construct(self, x0, x1, x2):
            y1 = ops.assign(x0, x1)
            y2 = ops.assign(x2, y1)
            y3 = ops.abs(y2)
            return y1, y2, y3

    parameter_idx = [0, 2]
    inputs = gen_inputs([(3, 4, 2)] * 3, [mindspore.bfloat16, mindspore.float16, mindspore.float32], parameter_idx)
    expects = get_output(Net(), inputs, parameter_idx, False)
    outputs = get_output(Net(), inputs, parameter_idx, True)
    compare_result = [np.allclose(e, o, 1e-5, 1e-5) for e, o in zip(expects, outputs)]
    assert False not in compare_result


def case3():
    class Net(Cell):
        def construct(self, x0, x1, x2):
            y1 = ops.assign(x0, x1)
            y2 = ops.assign(y1, x2)
            y3 = ops.abs(y2)
            return y1, y2, y3

    parameter_idx = [0]
    inputs = gen_inputs([(3, 4, 2)] * 3, [mindspore.float32] * 3, parameter_idx)
    expects = get_output(Net(), inputs, parameter_idx, False)
    outputs = get_output(Net(), inputs, parameter_idx, True)
    compare_result = [np.allclose(e, o, 1e-5, 1e-5) for e, o in zip(expects, outputs)]
    assert False not in compare_result


def case4():
    class Net(Cell):
        def construct(self, x0, x1, x2):
            y1 = ops.assign(x0, x1)
            y2 = ops.assign(y1, x2)
            y3 = ops.abs(y1)
            return y1, y2, y3

    parameter_idx = [0]
    inputs = gen_inputs([(3, 4, 2)] * 3, [mindspore.float32] * 3, parameter_idx)
    expects = get_output(Net(), inputs, parameter_idx, False)
    outputs = get_output(Net(), inputs, parameter_idx, True)
    compare_result = [np.allclose(e, o, 1e-5, 1e-5) for e, o in zip(expects, outputs)]
    assert False not in compare_result


def case5():
    class Net(Cell):
        def construct(self, x0, x1, x2):
            y1 = ops.assign(x0, x1)
            y2 = ops.div(x2, y1)
            y3 = ops.add(y1, y2)
            return y1, y3

    parameter_idx = [0]
    inputs = gen_inputs([(3, 4, 2)] * 3, [mindspore.float32] * 3, parameter_idx)
    expects = get_output(Net(), inputs, parameter_idx, False)
    outputs = get_output(Net(), inputs, parameter_idx, True)
    compare_result = [np.allclose(e, o, 1e-5, 1e-5) for e, o in zip(expects, outputs)]
    assert False not in compare_result


def case6():
    class Net(Cell):
        def construct(self, x, y, a, b, c):
            x = ops.abs(x)
            y = ops.abs(y)
            a = ops.abs(a)
            b = ops.abs(b)
            c = ops.abs(c)
            a -= b
            b %= a
            c /= b
            return y, a, c

    def run(enable_graph_kernel):
        np.random.seed(10)
        jit_level = "O1" if enable_graph_kernel else "O0"
        context.set_context(jit_config={"jit_level": jit_level})
        inputs = gen_inputs([(3, 4, 3, 1, 5, 5, 6, 1),
                             (3, 4, 3, 1, 5, 5, 6, 1),
                             (3, 3, 5, 3, 4, 1, 4),
                             (3, 3, 5, 3, 4, 1, 4),
                             (3, 3, 5, 3, 4, 1, 4)], [mindspore.float32] * 5, [])
        net = Net()
        out = net(*inputs)
        out_back = ops.functional.grad(net, (0, 1, 2, 3, 4))(*inputs)
        net.construct = mindspore.jit(net.construct)
        out_jit = net(*inputs)
        res = _output(out) + _output(out_back) + _output(out_jit)
        return res

    expects = run(False)
    outputs = run(True)
    compare_result = [np.allclose(e, o, 1e-5, 1e-5) for e, o in zip(expects, outputs)]
    assert False not in compare_result


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=4.00)
def test_inplace():
    """
    Feature: test inplace op
    Description: inplace and non-inplace fuse case
    Expectation: the result match with the expected result
    """
    context.set_context(mode=context.GRAPH_MODE)
    np.random.seed(1)
    case1()
    case2()
    case3()
    case4()
    case5()
    case6()
