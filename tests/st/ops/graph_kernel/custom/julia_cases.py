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
from mindspore.ops import DataType, CustomRegOp


class JuliaSingleOutputNet(Cell):
    def __init__(self, func, out_shapes, out_types, reg=None):
        super(JuliaSingleOutputNet, self).__init__()

        self.program = ops.Custom(func, out_shapes, out_types, "julia", reg_info=reg)

    def construct(self, x, y):
        return self.program(x, y)


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


def julia_single_output(func_name, bench, reg):
    shape = (4, 5)
    input_x = np.random.normal(0, 1, shape).astype(np.float32)
    input_y = np.random.normal(0, 1, shape).astype(np.float32)
    func_path = os.path.dirname(os.path.abspath(__file__)) + "/julia_test_files/"
    try:
        test = JuliaSingleOutputNet(func_path + func_name, (shape,), (mstype.float32,), reg)
        output = test(Tensor(input_x), Tensor(input_y))[0]
    except Exception as e:
        raise e
    assert np.allclose(bench(input_x, input_y), output.asnumpy(), 0.001, 0.001)


cpu_info = CustomRegOp() \
    .input(0, "x1") \
    .input(1, "x2") \
    .output(0, "y") \
    .dtype_format(DataType.None_None, DataType.None_None, DataType.None_None) \
    .target("CPU") \
    .get_op_info()


@pytest.mark.level2
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
        julia_single_output("add.jl:Add:foo!", add, cpu_info)


@pytest.mark.level2
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
        julia_single_output("sub.jl:Sub:foo!", sub, cpu_info)
