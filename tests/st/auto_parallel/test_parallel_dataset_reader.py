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
"""Test runner for pipeline parallelism with dataset sharding strategy and distributed checkpoints.

This module serves as a test execution wrapper that launches distributed tests for pipeline parallelism
with dataset-level sharding strategies and distributed checkpoint management. It uses the msrun
distributed launcher to execute the actual test implementation across 8 Ascend devices.

The test verifies that:
- Pipeline parallelism with 2 stages functions correctly across 8 devices
- Dataset sharding strategies are properly applied during training
- Distributed checkpoints can be saved and loaded from different pipeline configurations
- Inference results are numerically consistent across different training runs
- Graph optimizations (e.g., Broadcast operations) are correctly applied

Execution Details:
- Launcher: msrun (MindSpore distributed launcher)
- Worker configuration: 8 workers with 8 local workers
- Master address: 127.0.0.1 (localhost for single machine)
- Master port: 10809
- Logging: Directed to ./parallel_batchmatmul_high_dim log directory
- Test framework: pytest with verbose output

The actual test logic is implemented in parallel_dataset_reader.py, which contains:
- WithLossCell: Network wrapper combining backbone with loss function
- ParallelBiasAddNet: Distributed bias-add network cell
- ParallelMulNet: Distributed multiplication cell
- PipeDatasetStrategy: Multi-stage pipeline network
- mindspore_pipeline_predict: Distributed inference function
- test_parallel_dataset_reader_dataset_strategy_pipeline_002: Main test function

Testing Phases:
1. Train with pipeline1 configuration and save distributed checkpoints
2. Train with pipeline2 configuration and dataset speed-up optimization
3. Load both checkpoint configurations and verify inference consistency
4. Validate dataset optimization IR graphs contain Broadcast operations
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_parallel_dataset_reader_dataset_strategy_pipeline_002():
    '''
    Feature: parallel dataset reader.
    Description: Test parallel dataset reader.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./parallel_dataset_reader pytest -s -v "
                    "parallel_dataset_reader.py::test_parallel_dataset_reader_dataset_strategy_pipeline_002")
    assert ret == 0
