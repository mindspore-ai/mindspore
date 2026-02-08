
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
The tests of mindspore, used to test communication for tcpstore.
"""
import os
from tests.mark_utils import arg_mark
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_tcp_store():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        "--master_port=10668 --join=True pytest -s test_tcp_store.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_hccl_mint_tcp_store1():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        r"cp  test_tcp_store.py test_tcp_store1.py && "\
        r"sed -i 's/mindspore\.mint\.distributed\.distributed/mindspore.ops.communication/g' "\
        r"test_tcp_store1.py && msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        r"--master_port=10668 --join=True pytest -s test_tcp_store1.py"
    )
    assert return_code == 0
