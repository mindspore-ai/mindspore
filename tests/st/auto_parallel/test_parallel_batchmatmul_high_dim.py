# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Test runner for high-dimensional batch matrix multiplication with N-dimensional tensor parallelism.

This module serves as a test execution wrapper that launches distributed tests for high-dimensional
BatchMatMul operations with N-dimensional tensor parallelism (ND-TP) optimization. It uses the msrun
distributed launcher to execute the actual test implementation across 8 devices on Ascend hardware.

The test verifies that:
- BatchMatMul operations correctly handle high-dimensional tensors
- N-dimensional tensor parallelism optimization is properly applied
- Distributed execution across 8 Ascend devices functions correctly
- Complex sharding strategies with multi-dimensional device mesh work as expected

Execution Details:
- Launcher: msrun (MindSpore distributed launcher)
- Worker configuration: 8 workers with 8 local workers
- Master address: 127.0.0.1 (localhost for single machine)
- Master port: 10809
- Logging: Directed to ./parallel_batchmatmul_high_dim log directory
- Test framework: pytest with verbose output

The actual test logic is implemented in parallel_batchmatmul_high_dim.py, which contains
the BatchMatMulCell network, layout definitions, and test assertions.
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_batchmatmul_enable_nd_tp_002():
    '''
    Feature: bmm nd tp.
    Description: Test bmm nd tp.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./parallel_batchmatmul_high_dim pytest -s -v "
                    "parallel_batchmatmul_high_dim.py::test_parallel_batchmatmul_enable_nd_tp_002")
    assert ret == 0
