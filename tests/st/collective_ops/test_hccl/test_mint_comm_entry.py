
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
"""
The tests of mindspore, used to test communication for mint.distributed.
"""
import os
from tests.mark_utils import arg_mark
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_distributed.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_test_gather_scatter_tensor():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_gather_scatter_tensor.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_infer_error():
    """
    Feature: msrun 8P case
    Description:msrun 8P case
    Expectation: success
    """
    return_code = os.system(
        "export MS_DEV_HOST_BLOCKING_RUN=1; msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        " --master_port=10666 --join=True pytest -s test_distributed_infer_except.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_test_gather_scatter_tensor_except():
    """
    Feature: msrun 8P case
    Description:msrun 8P case
    Expectation: success
    """
    return_code = os.system(
        "export MS_DEV_HOST_BLOCKING_RUN=1; msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        " --master_port=10666 --join=True pytest -s test_gather_scatter_tensor_except.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_object_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_comm_object.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_init_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 --join=True "\
        "pytest -s test_comm_init.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_comm_func_ops1():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        r"cp  test_distributed.py test_distributed1.py && "\
        r"sed -i 's/mindspore\.mint\.distributed\.distributed/mindspore.ops.communication/g' "\
        r"test_distributed1.py && msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        r"--master_port=10666 --join=True pytest -s test_distributed1.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_object_ops1():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        r"cp  test_comm_object.py test_comm_object1.py && "\
        r"sed -i 's/mindspore\.mint\.distributed\.distributed/mindspore.ops.communication/g' "\
        r"test_comm_object1.py && msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        r"--master_port=10666 --join=True pytest -s test_comm_object1.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_init_ops1():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        r"cp  test_comm_init.py test_comm_init1.py && "\
        r"sed -i 's/mindspore\.mint\.distributed\.distributed/mindspore.ops.communication/g' "\
        r"test_comm_init1.py && msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        r"--master_port=10666 --join=True pytest -s test_comm_init1.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_init_ops2():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        "--master_port=10666 --join=True pytest -s test_comm_func_noninplace.py"
    )
    assert return_code == 0
