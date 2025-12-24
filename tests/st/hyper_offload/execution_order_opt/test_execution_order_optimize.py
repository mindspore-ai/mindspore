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
"Test whether hyper offload auto optimize with can decrease autual used memory"
import os
import re
import subprocess
import pytest
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore import jit, Tensor, Parameter
from mindspore._extends.parse import compile_config
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype

def extract_memory_usage(filename):
    pattern = r'Actual peak memory usage \(with fragments\): (\d+)M'
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                memory_usage = int(match.group(1))
                return memory_usage
        return None

def get_max_memory(file_name, log_file_name):
    # Clear compile cache folder and log files
    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    os.environ['MS_DEV_RUNTIME_CONF'] = "memory_statistics:True"
    os.environ['MS_DEV_HYPER_OFFLOAD'] = "1"

    cmd = "python {} > {} 2>&1".format(file_name, log_file_name)
    subprocess.check_output(cmd, shell=True)
    assert os.path.exists(log_file_name)

    memory_useage = extract_memory_usage(log_file_name)

    os.remove(log_file_name)
    os.environ['MS_DEV_HYPER_OFFLOAD'] = "0"
    os.environ['MS_DEV_RUNTIME_CONF']=""
    return memory_useage


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_base():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Test whether hyper offload auto optimize with can decrease autual used memory.
    Expectation: success.
    """
    max_memory = get_max_memory("run_net_base.py", "test_net_base_result.txt")
    assert max_memory < 160


def test_net_parameter():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Test whether hyper offload auto optimize with can decrease autual used memory.
    Expectation: success.
    """
    max_memory = get_max_memory("run_net_parameter.py", "test_net_parameter_result.txt")
    assert max_memory < 160


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_net_all():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: Test whether hyper offload auto optimize with can decrease autual used memory.
    Expectation: success.
    """
    max_memory = get_max_memory("run_net_all.py", "test_net_parameter_all.txt")
    assert max_memory < 160


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_with_remote():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: auto_offload should validate input should only be all, weight and activaction.
    Expectation: RuntimeError.
    """
    class TestNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = Parameter(Tensor(np.ones([2, 2]), dtype=mstype.float32), device="Remote", name="p")

        @jit(auto_offload="weight")
        def construct(self, x):
            a = self.p + x
            ops.assign(self.p, a)

    os.environ['MS_DEV_HYPER_OFFLOAD'] = "1"
    origin_prefetch_distance = compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = 0
    net = TestNet()
    net(Tensor(np.ones([2, 2]), dtype=mstype.float32))
    assert np.all(net.p.value().asnumpy() == np.array([[2, 2], [2, 2]]))
    os.environ['MS_DEV_HYPER_OFFLOAD'] = ""
    compile_config.HYPER_OFFLOAD_PREFETCH_DISTANCE = origin_prefetch_distance


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_wrong_auto_offload_input():
    """
    Feature: Hyper offload auto optimize with execution order.
    Description: auto_offload should validate input should only be all, weight and activaction.
    Expectation: RuntimeError.
    """
    @jit(auto_offload="wrong_input")
    def foo(x):
        return x + 1

    with pytest.raises(RuntimeError) as e:
        foo(Tensor([1, 2, 3]))
    assert "auto_offload should only be value of all, weight or activaction but got" in str(e.value)
