# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import context


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_lccl_allreduce():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_reduce 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_allreduce.py")
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_lccl_allgather():
    """
    Feature: lccl operator test.
    Description: msrun lccl all_gather 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_allgather.py")
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_lccl_reducescatter():
    """
    Feature: lccl operator test.
    Description: msrun lccl reduce_scatter 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True "
                            "pytest -s test_lccl_reduce_scatter.py")
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_single
def test_lccl_broadcast():
    """
    Feature: lccl operator test.
    Description: msrun lccl broadcast 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_broadcast.py")
    assert return_code == 0


@pytest.mark.level2
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_lccl_matmul_allreduce():
    """
    Feature: lccl MatMulAllReduce fustion operator test.
    Description: lccl MatMulAllReduce 8P case.
    Expectation: success
    """
    os.environ['MS_ENABLE_LCCL'] = str(1)
    os.environ['MS_ENABLE_INTERNAL_KERNELS'] = 'on'
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_lccl_matmul_allreduce.py")
    assert return_code == 0
