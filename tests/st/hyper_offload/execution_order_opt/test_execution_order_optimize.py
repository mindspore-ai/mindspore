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
"""Tests for HyperOffload execution order optimization memory reduction."""

import os
import re
import subprocess

import numpy as np
import pytest
from mindspore import Parameter, Tensor, jit, nn, ops
import mindspore.common.dtype as mstype
from mindspore.graph._parse import compile_config

from tests.mark_utils import arg_mark


_MEMORY_PATTERN = re.compile(r"Actual peak memory usage \(with fragments\): (\d+)M")
_RUNTIME_CONF_KEY = "MS_DEV_RUNTIME_CONF"
_HYPER_OFFLOAD_KEY = "MS_DEV_HYPER_OFFLOAD"


def extract_memory_usage(filename):
    """Extract peak memory usage in MB from log file."""
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            match = _MEMORY_PATTERN.search(line)
            if match:
                return int(match.group(1))
    return None


def _restore_env_var(env_name, old_value):
    """Restore env var to previous value."""
    if old_value is None:
        os.environ.pop(env_name, None)
    else:
        os.environ[env_name] = old_value


def _run_and_get_max_memory(file_name, log_file_name, enable_hyper_offload):
    """Run script and parse maximum memory usage from generated logs."""
    old_runtime_conf = os.environ.get(_RUNTIME_CONF_KEY)
    old_hyper_offload = os.environ.get(_HYPER_OFFLOAD_KEY)
    try:
        if os.path.exists(log_file_name):
            os.remove(log_file_name)

        os.environ[_RUNTIME_CONF_KEY] = "memory_statistics:True"
        if enable_hyper_offload:
            os.environ[_HYPER_OFFLOAD_KEY] = "1"
        else:
            os.environ.pop(_HYPER_OFFLOAD_KEY, None)

        cmd = "python {} > {} 2>&1".format(file_name, log_file_name)
        subprocess.check_output(cmd, shell=True)
        assert os.path.exists(log_file_name)
        return extract_memory_usage(log_file_name)
    finally:
        if os.path.exists(log_file_name):
            os.remove(log_file_name)
        _restore_env_var(_RUNTIME_CONF_KEY, old_runtime_conf)
        _restore_env_var(_HYPER_OFFLOAD_KEY, old_hyper_offload)


def get_max_memory(file_name, log_file_name):
    """Get maximum memory usage with HyperOffload enabled."""
    return _run_and_get_max_memory(file_name, log_file_name, True)


def get_max_memory_baseline(file_name, log_file_name):
    """Get baseline maximum memory usage without HyperOffload."""
    return _run_and_get_max_memory(file_name, log_file_name, False)


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=58145)
def test_net_base():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify base net can reduce memory by hyper offload auto optimize.
    Expectation: success.
    """
    max_memory = get_max_memory("run_net_base.py", "test_net_base_result.txt")
    assert max_memory < 160


def test_net_parameter():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify parameter net with HyperOffload reduces memory compared to baseline.
    Expectation: success.
    """
    max_memory_baseline = get_max_memory_baseline("run_net_parameter_base.py", "parameter_base.txt")
    max_memory = get_max_memory("run_net_parameter.py", "test_net_parameter_result.txt")
    assert max_memory < max_memory_baseline


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=0.01)
def test_net_all():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify all-mode memory usage is reduced by hyper offload auto optimize.
    Expectation: success.
    """
    max_memory = get_max_memory("run_net_all.py", "test_net_parameter_all.txt")
    assert max_memory < 160


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=0.01)
def test_net_all_trace():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify all-mode trace execution reduces memory with hyper offload.
    Expectation: success.
    """
    max_memory = get_max_memory("run_net_all_trace.py", "test_net_parameter_all_trace.txt")
    assert max_memory < 400


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_net_memory_peak():
    """
    Feature: Memory peak based hyper offload strategy.
    Description: Test memory peak based strategy limits device memory usage.
    Expectation: Peak memory less than 300M.
    """
    max_memory = get_max_memory("run_net_memory_peak.py", "test_net_memory_peak_result.txt")
    assert max_memory < 300


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_parameter_with_remote():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify remote parameter assignment works in auto_offload weight mode.
    Expectation: RuntimeError.
    """

    class TestNet(nn.Cell):
        """Simple remote parameter update network."""

        def __init__(self):
            super().__init__()
            self.p = Parameter(Tensor(np.ones([2, 2]), dtype=mstype.float32), device="Remote", name="p")

        @jit(auto_offload="weight")
        def construct(self, x):
            updated_value = self.p + x
            ops.assign(self.p, updated_value)

    old_hyper_offload = os.environ.get(_HYPER_OFFLOAD_KEY)
    original_prefetch_distance = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    try:
        os.environ[_HYPER_OFFLOAD_KEY] = "1"
        compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = 0
        net = TestNet()
        net(Tensor(np.ones([2, 2]), dtype=mstype.float32))
        assert np.all(net.p.value().asnumpy() == np.array([[2, 2], [2, 2]]))
    finally:
        compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = original_prefetch_distance
        _restore_env_var(_HYPER_OFFLOAD_KEY, old_hyper_offload)


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=0.01)
def test_wrong_auto_offload_input():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify invalid auto_offload value throws ValueError.
    Expectation: RuntimeError.
    """
    with pytest.raises(ValueError) as error_info:

        @jit(auto_offload="wrong_input")
        def foo(x):
            return x + 1

        foo(Tensor([1, 2, 3]))

    assert "For 'auto_offload', the value of 'auto_offload' must be one of " in str(error_info.value)


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    mem_peak=4.00)
def test_parameter_with_too_long_prefetch_distance():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Verify overlarge prefetch distance does not cause crash.
    Expectation: success.
    """

    class TestNet(nn.Cell):
        """Simple remote parameter update network."""

        def __init__(self):
            super().__init__()
            self.p = Parameter(Tensor(np.ones([2, 2]), dtype=mstype.float32), device="Remote", name="p")

        @jit(auto_offload="weight")
        def construct(self, x):
            updated_value = self.p + x
            ops.assign(self.p, updated_value)

    old_hyper_offload = os.environ.get(_HYPER_OFFLOAD_KEY)
    original_select_distance = compile_config.HYPER_OFFLOAD_SELECT_DISTANCE
    original_select_num = compile_config.HYPER_OFFLOAD_SELECT_NUM
    original_prefetch_distance = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    try:
        os.environ[_HYPER_OFFLOAD_KEY] = "1"
        compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = 100
        compile_config.HYPER_OFFLOAD_SELECT_NUM = 100
        compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = 200
        net = TestNet()
        net(Tensor(np.ones([2, 2]), dtype=mstype.float32))
        assert np.all(net.p.value().asnumpy() == np.array([[2, 2], [2, 2]]))
    finally:
        compile_config.HYPER_OFFLOAD_SELECT_DISTANCE = original_select_distance
        compile_config.HYPER_OFFLOAD_SELECT_NUM = original_select_num
        compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = original_prefetch_distance
        _restore_env_var(_HYPER_OFFLOAD_KEY, old_hyper_offload)
