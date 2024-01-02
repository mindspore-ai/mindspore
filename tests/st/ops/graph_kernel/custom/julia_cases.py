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

import os
import platform
import numpy as np
import pytest
from mindspore import context, Tensor
from mindspore.common import dtype as mstype
from mindspore.nn import Cell
import mindspore.ops as ops


class JuliaTwoInputsNet(Cell):
    def __init__(self, func, out_shapes, out_types, reg=None):
        super(JuliaTwoInputsNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "julia", reg_info=reg)

    def construct(self, x, y):
        return self.program(x, y)


class JuliaOneInputNet(Cell):
    def __init__(self, func, out_shapes, out_types, reg=None):
        super(JuliaOneInputNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "julia", reg_info=reg)

    def construct(self, x):
        return self.program(x)


def add(x, y):
    """
    function add for benchmark
    """
    return x + y


def sub(x, y):
    """
    function sub for benchmark
    """
    return x - y


def matmul(x, y):
    """
    function matmul for benchmark
    """
    return np.matmul(x, y)


def reducesum(x, axis=0, keepdims=True):
    return np.sum(x, axis=axis, keepdims=keepdims)


def multiout(a, b):
    return a + b, a - b


def julia_elemwise_test(func_name, bench):
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float32)
    input_y = np.random.normal(0, 1, shape).astype(np.float32)
    func_path = os.path.dirname(os.path.abspath(__file__)) + "/julia_test_files/"
    try:
        test = JuliaTwoInputsNet(func_path + func_name, (shape,), (mstype.float32,))
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        raise e
    assert np.allclose(bench(input_x, input_y), output.asnumpy(), 0.001, 0.001)


def julia_matmul_test(func_name, bench):
    shape1 = (2, 3)
    shape2 = (3, 4)
    shape3 = (2, 4)
    input_x = np.random.normal(0, 1, shape1).astype(np.float32)
    input_y = np.random.normal(0, 1, shape2).astype(np.float32)
    func_path = os.path.dirname(os.path.abspath(__file__)) + "/julia_test_files/"
    try:
        test = JuliaTwoInputsNet(func_path + func_name, (shape3,), (mstype.float32,))
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        raise e
    assert np.allclose(bench(input_x, input_y), output.asnumpy(), 0.001, 0.001)


def julia_reducesum_test(func_name, bench):
    shape1 = (2, 3, 4)
    input_x = np.random.normal(0, 1, shape1).astype(np.float32)
    expect = bench(input_x, 1)
    func_path = os.path.dirname(os.path.abspath(__file__)) + "/julia_test_files/"
    try:
        test = JuliaOneInputNet(func_path + func_name, (expect.shape,), (mstype.float32,))
        output = test(Tensor(input_x))[0]
    except Exception as e:
        raise e
    assert np.allclose(expect, output.asnumpy(), 0.001, 0.001)


def julia_multiout_test(func_name, bench):
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float32)
    input_y = np.random.normal(0, 1, shape).astype(np.float32)
    func_path = os.path.dirname(os.path.abspath(__file__)) + "/julia_test_files/"
    try:
        test = JuliaTwoInputsNet(func_path + func_name, (shape, shape,), (mstype.float32, mstype.float32,))
        output1 = test(Tensor(input_x), Tensor(input_y))[0]
        output2 = test(Tensor(input_x), Tensor(input_y))[1]
    except Exception as e:
        raise e
    expect1, expect2 = bench(input_x, input_y)
    assert np.allclose(expect1, output1.asnumpy(), 0.001, 0.001)
    assert np.allclose(expect2, output2.asnumpy(), 0.001, 0.001)


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia_single_output_cpu_add():
    """
    Feature: custom julia operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-write xxx.jl, custom operator launches xxx.jl
    Expectation: nn result matches numpy result
    """
    system = platform.system()
    if system != 'Linux':
        pass
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        julia_elemwise_test("add.jl:Add:foo!", add)


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia_single_output_cpu_sub():
    """
    Feature: custom julia operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-write xxx.jl, custom operator launches xxx.jl
    Expectation: nn result matches numpy result
    """
    system = platform.system()
    if system != 'Linux':
        pass
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        julia_elemwise_test("sub.jl:Sub:foo!", sub)


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia_single_output_cpu_matmul():
    """
    Feature: custom julia operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-write xxx.jl, custom operator launches xxx.jl
    Expectation: nn result matches numpy result
    """
    system = platform.system()
    if system != 'Linux':
        pass
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        julia_matmul_test("matmul.jl:Matmul:foo!", matmul)


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia_single_output_cpu_reducesum():
    """
    Feature: custom julia operator, multiple inputs, single output, CPU, GRAPH_MODE
    Description: pre-write xxx.jl, custom operator launches xxx.jl
    Expectation: nn result matches numpy result
    """
    system = platform.system()
    if system != 'Linux':
        pass
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        julia_reducesum_test("reducesum.jl:ReduceSum:foo!", reducesum)


@pytest.mark.level3
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_julia_multi_output_cpu():
    """
    Feature: custom julia operator, multiple inputs, multi output, CPU, GRAPH_MODE
    Description: pre-write xxx.jl, custom operator launches xxx.jl
    Expectation: nn result matches numpy result
    """
    system = platform.system()
    if system != 'Linux':
        pass
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
        julia_multiout_test("multi_output.jl:MultiOutput:foo!", multiout)
