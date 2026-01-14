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

"""test augassign memory use."""

import os
import re
import subprocess
import numpy as np
import mindspore as ms
from mindspore import jit
from mindspore.graph._parse import compile_config
from tests.mark_utils import arg_mark

match_dyn_mem = re.compile(r'Used peak memory usage \(without fragments\)\: (.*?)M', re.S)


def get_max(mem_uses):
    max_mem = 0
    for i in mem_uses:
        max_mem = max(max_mem, int(i))
    return max_mem


def run_testcase(testcase_name):
    log_filename = testcase_name + ".log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    assert not os.path.exists(log_filename)

    cmd = ("export GLOG_v=1; export MS_DEV_RUNTIME_CONF=\"memory_statistic:True\"; pytest -s "
           "test_augassign_memory_use.py::" + testcase_name + " > " + log_filename + " 2>&1")
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_filename)
    # pylint: disable=W1514
    with open(log_filename, "r") as f:
        data = f.read()
    mem_uses = re.findall(match_dyn_mem, data)
    max_mem = get_max(mem_uses)
    if os.path.exists(log_filename):
        os.remove(log_filename)
    return max_mem


@jit(backend="ms_backend")
def add_func(x, y):
    x += y
    return x


def test_add_memory():
    """
    Feature: Support augassign inplace.
    Description: Test the memory usage of assignment statements.
    Expectation: Run success.
    """

    try:
        compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '0'
        shape = (1000, 1000)
        x = ms.Tensor(np.random.randn(*shape).astype(np.float32))
        y = ms.Tensor(np.random.randn(*shape).astype(np.float32))
        add_func(x, y)
    finally:
        compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '1'


def test_inplace_add_memory():
    """
    Feature: Support augassign inplace.
    Description: Test the memory usage of assignment statements.
    Expectation: Run success.
    """

    shape = (1000, 1000)
    x = ms.Tensor(np.random.randn(*shape).astype(np.float32))
    y = ms.Tensor(np.random.randn(*shape).astype(np.float32))
    add_func(x, y)


@jit(backend="ms_backend")
def mul_func(x, y):
    x *= y
    return x


def test_mul_memory():
    """
    Feature: Support augassign inplace.
    Description: Test the memory usage of assignment statements.
    Expectation: Run success.
    """

    try:
        compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '0'
        shape = (1000, 1000)
        x = ms.Tensor(np.random.randn(*shape).astype(np.float32))
        y = ms.Tensor(np.random.randn(*shape).astype(np.float32))
        mul_func(x, y)
    finally:
        compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '1'


def test_inplace_mul_memory():
    """
    Feature: Support augassign inplace.
    Description: Test the memory usage of assignment statements.
    Expectation: Run success.
    """

    shape = (1000, 1000)
    x = ms.Tensor(np.random.randn(*shape).astype(np.float32))
    y = ms.Tensor(np.random.randn(*shape).astype(np.float32))
    mul_func(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=12.00)
def test_augassign_inplace_add_memory_use():
    """
    Feature: Support augassign inplace.
    Description: Test the memory usage of assignment statements.
    Expectation: Run success.
    """
    add_memory = run_testcase("test_add_memory")
    inplace_add_memory = run_testcase("test_inplace_add_memory")
    assert add_memory >= inplace_add_memory


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=12.00)
def test_augassign_inplace_mul_memory_use():
    """
    Feature: Support augassign inplace.
    Description: Test the memory usage of assignment statements.
    Expectation: Run success.
    """
    mul_memory = run_testcase("test_mul_memory")
    inplace_mul_memory = run_testcase("test_inplace_mul_memory")
    assert mul_memory >= inplace_mul_memory
