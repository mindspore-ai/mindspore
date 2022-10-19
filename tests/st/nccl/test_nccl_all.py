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
import os
import pytest


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_lenet():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_lenet.py")
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_all_reduce_op():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_all_reduce_op.py")
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_all_gather_op():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_all_gather_op.py")
    assert return_code == 0


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_reduce_scatter_op():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_reduce_scatter_op.py")
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_broadcast_op():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_broadcast_op.py")
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_send_recv_op():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_send_recv_op.py")
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_all_to_all_op():
    return_code = os.system("mpirun -n 8 pytest -s test_nccl_all_to_all_op.py")
    assert return_code == 0

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_single
def test_nccl_neighbor_exchange_op():
    """
    Feature: NeighborExchange GPU operator
    Description: see details in test_nccl_neighbor_exchange_op.py
    Expectation: success, return_code==0
    """
    return_code = os.system(
        "mpirun -n 8 pytest -s test_nccl_neighbor_exchange_op.py")
    assert return_code == 0
