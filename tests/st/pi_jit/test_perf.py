# Copyright 2023 Huawei Technologies Co., Ltd
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
"""run perf statistic test"""
import pytest
from mindspore import Tensor, jit, context


@jit(mode="PIJit", jit_config={"perf_statistics": True})
def perf_statistic_simple(a, b):
    return a + b


@jit(mode="PIJit", jit_config={"perf_statistics": True, "PERF_STATISTICS_SCALE_10000X": -9900})
def perf_statistic_complex(a, b):
    a = a + b
    b = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    b = a + b
    a = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    b = a + b
    a = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    b = a + b
    a = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    return a


@jit(mode="PIJit", jit_config={"STATIC_GRAPH_BYTECODE_MIN": 8})
def perf_bytecode_simple(a, b):
    return a + b


@jit(mode="PIJit", jit_config={"STATIC_GRAPH_BYTECODE_MIN": 8})
def perf_bytecode_complex(a, b):
    a = a + b
    b = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    b = a + b
    a = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    b = a + b
    a = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    b = a + b
    a = a - b
    a = b + 1
    b = a ** 2
    a = a // 2
    return a


@pytest.mark.skip
@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('func_param', [(perf_statistic_simple, False), (perf_statistic_complex, True),
                                        (perf_bytecode_simple, False), (perf_bytecode_complex, True)])
def test_perf_statistic_case(func_param):
    """
    Feature: Method Perf Testing
    Description: Test performance function to check whether the graph is executed as expected.
    Expectation: The result of the case should check whether the graph is executed as expected.
                 'perf_statistics' flag is used to make statistics for which one of graph and pynative can run faster.
                 'PERF_STATISTICS_SCALE_10000X' flag is used to set the preference of execution as graph.
                 'STATIC_GRAPH_BYTECODE_MIN' flag is used to limit the minimal bytecode size when executing as graph.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    func = func_param[0]
    param = func_param[1]
    a = Tensor([1])
    b = Tensor([2])
    func(a, b)
    func(a, b)
    func(a, b)
    if param is True:
        assert func.__globals__[func.__code__]
    else:
        assert not func.__globals__[func.__code__]
