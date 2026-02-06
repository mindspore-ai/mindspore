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
"""Test runner for communication group reuse and custom collective operations in distributed training.

This module serves as a test execution wrapper that launches distributed tests for custom communication
group creation and reuse in MindSpore's distributed training framework. It uses the msrun distributed
launcher to execute the actual test implementation across 8 Ascend devices.

The test verifies that:
- Custom communication groups can be created and reused across multiple operations
- Group-based collective operations (AllReduce, Broadcast) function correctly
- Distributed training with custom groups produces numerically consistent results
- Standalone and parallel training modes produce equivalent outputs
- IR graph verification confirms proper communication group assignment

Execution Details:
- Launcher: msrun (MindSpore distributed launcher)
- Worker configuration: 8 workers with 8 local workers
- Master address: 127.0.0.1 (localhost for single machine)
- Master port: 10809
- Logging: Directed to ./parallel_batchmatmul_high_dim log directory
- Test framework: pytest with verbose output

The actual test logic is implemented in parallel_group_reuse.py, which contains:
- CompareBase: Numerical comparison utility for validation
- Net1: Neural network with Add and MatMul operations supporting custom sharding
- check_ir(): IR graph verification function
- save_graphs(): Graph saving configuration utility
- test_parallel_creat_group_reuse_001: Main test function with 5-phase workflow

Testing Phases:
1. Execute standalone training without distributed communication
2. Execute parallel training with custom communication groups
3. Compare checkpoint files between standalone and parallel modes
4. Verify numerical equivalence of model weights
5. Validate IR graphs contain correct communication group operations
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_creat_group_reuse_001():
    '''
    Feature: Parallel create group reuse.
    Description: Test parallel create group reuse.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./parallel_group_reuse pytest -s -v "
                    "parallel_group_reuse.py::test_parallel_creat_group_reuse_001")
    assert ret == 0
