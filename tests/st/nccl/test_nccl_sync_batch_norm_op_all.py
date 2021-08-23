# Copyright 2021 Huawei Technologies Co., Ltd
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
import pytest

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_sync_batch_norm_1():
    cmd_str = "mpirun -n 4 pytest -s test_nccl_sync_batch_norm_op.py::test_sync_batch_norm_forward_fp32_graph"
    return_code = os.system(cmd_str)
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_sync_batch_norm_2():
    cmd_str = "mpirun -n 4 pytest -s test_nccl_sync_batch_norm_op.py::test_sync_batch_norm_forward_fp16_pynative"
    return_code = os.system(cmd_str)
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_sync_batch_norm_3():
    cmd_str = "mpirun -n 1 pytest -s test_nccl_sync_batch_norm_op.py::test_sync_batch_norm_backwards_fp32_graph"
    return_code = os.system(cmd_str)
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_sync_batch_norm_4():
    cmd_str = "mpirun -n 1 pytest -s test_nccl_sync_batch_norm_op.py::test_sync_batch_norm_backwards_fp16_pynative"
    return_code = os.system(cmd_str)
    assert return_code == 0
