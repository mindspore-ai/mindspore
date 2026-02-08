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
"""Test runner for zero-bubble pipeline parallelism strategy in distributed training.

This module serves as a test execution wrapper that launches distributed tests for the zero-bubble
pipeline strategy (Zero-Bubble-V) in MindSpore. It uses the msrun distributed launcher to execute
the actual test implementation across 8 Ascend devices.

The test verifies that:
- Zero-bubble pipeline strategy achieves minimal pipeline bubble
- Multi-stage pipeline parallelism with 4 stages functions correctly
- Checkpoint transformation and consolidation work with zero-bubble scheduling
- Distributed checkpoints can be loaded and used for inference
- Numerical consistency is maintained across different pipeline configurations

Zero-Bubble-V Strategy Features Tested:
- Optimal scheduling for multi-stage pipeline parallelism
- Minimization of idle time (bubble) in pipeline stages
- Efficient microbatch scheduling across 4 stages
- Memory-efficient checkpoint transformation
- Asynchronous communication and computation overlap

Execution Details:
- Launcher: msrun (MindSpore distributed launcher)
- Worker configuration: 8 workers with 8 local workers
- Master address: 127.0.0.1 (localhost for single machine)
- Master port: 10809
- Logging: Directed to ./zero_bubble log directory
- Test framework: pytest with verbose output

The actual test logic is implemented in zero_bubble_v.py, which contains:
- WithLossCell: Network wrapper combining backbone with loss function
- MatMulNet: Matrix multiplication operations network
- StageNet: Multi-stage network with 4 micro-stages for pipeline parallelism
- Helper functions: check_checkpoint_file_by_rank, predict_each_rank, _count_unequal_element, allclose_nparray
- test_pipeline_zero_bubble_v_001: Main comprehensive test with 8 phases

Test Configuration:
- Device count: 8 devices (2 devices per pipeline stage)
- Pipeline stages: 4
- Dataset: FakeData with 256 samples, batch size 32
- Network shape: (32, 3, 16, 16) input, multiple MatMul operations
- Checkpoint format: Unified safetensors with consolidation
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_pipeline_zero_bubble_v():
    '''
    Feature: Pipeline zero bubble v.
    Description: Test zero bubble v.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./zero_bubble pytest -s -v "
                    "zero_bubble_v.py::test_pipeline_zero_bubble_v_001")
    assert ret == 0
