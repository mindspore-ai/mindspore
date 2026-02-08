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
"""Test runner for pipeline interleaving schedulers (GPipe and 1F1B) in distributed training.

This module serves as a test execution wrapper that launches distributed tests for pipeline
parallelism with different interleaving scheduling strategies in MindSpore. It uses the msrun
distributed launcher to execute the actual test implementation across 8 Ascend devices.

The test verifies that:
- Pipeline interleaving with GPipe scheduler functions correctly
- Pipeline interleaving with 1F1B (One Forward One Backward) scheduler works as expected
- Multi-stage pipeline parallelism with interleaved scheduling improves GPU utilization
- Both scheduler types handle distributed training across 2 pipeline stages
- Checkpoint saving/loading works correctly with pipeline interleaving

Pipeline Schedulers Tested:
- GPipe: Sequential processing of all microbatches with interleaving
  * Lower memory overhead but higher pipeline bubble
  * All forward passes followed by all backward passes
  
- 1F1B: Alternating forward and backward passes with multiple in-flight microbatches
  * Higher memory requirement for microbatch buffering
  * Reduced pipeline bubble with better GPU utilization

Execution Details:
- Launcher: msrun (MindSpore distributed launcher)
- Worker configuration: 8 workers with 8 local workers
- Master address: 127.0.0.1 (localhost for single machine)
- Master port: 10809
- Logging: Directed to ./parallel_batchmatmul_high_dim log directory
- Test framework: pytest with verbose output

The actual test logic is implemented in pipeline_interleave.py, which contains:
- Net: LeNet-style CNN with 2 conv layers and 3 dense layers
- setup_function: Environment initialization
- test_pipeline_interleave_001: Test function for GPipe scheduler
- test_pipeline_interleave_004: Test function for 1F1B scheduler with 4 microbatches

Test Configuration:
- Device count: 8 devices (4 devices per pipeline stage)
- Pipeline stages: 2
- Conv1 layer sharding: Distributed across 4 devices in batch dimension
- Input shape: (batch_size, 1, 32, 32)
- Dataset: FakeData with 256 samples, batch size 256
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_pipeline_interleave_001():
    '''
    Feature: Pipeline interleave.
    Description: Test pipeline interleave case 001.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./pipeline_interleave_001 pytest -s -v "
                    "pipeline_interleave.py::test_pipeline_interleave_001")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_pipeline_interleave_004():
    '''
    Feature: Pipeline interleave.
    Description: Test pipeline interleave case 001.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./pipeline_interleave_004 pytest -s -v "
                    "pipeline_interleave.py::test_pipeline_interleave_004")
    assert ret == 0
