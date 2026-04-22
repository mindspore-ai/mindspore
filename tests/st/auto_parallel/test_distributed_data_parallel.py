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

import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_full_batch_DDP_without_bucket_rebuilt():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP single fullbatch training without bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_full_batch_DDP_without_bucket_rebuilt"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_full_batch_DDP_with_bucket_rebuilt():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP single fullbatch training with bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_full_batch_DDP_with_bucket_rebuilt"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_accumulate_batch_DDP_with_bucket_rebuilt():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP accumulated batch training with bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_accumulate_batch_DDP_with_bucket_rebuilt"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_accumulate_batch_DDP_without_bucket_rebuilt():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP accumulated batch training without bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_accumulate_batch_DDP_without_bucket_rebuilt"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_full_batch_DDP_without_bucket_rebuilt_cpp():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP single fullbatch training without bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_full_batch_DDP_without_bucket_rebuilt_cpp"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_full_batch_DDP_with_bucket_rebuilt_cpp():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP single fullbatch training with bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_full_batch_DDP_with_bucket_rebuilt_cpp"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_accumulate_batch_DDP_with_bucket_rebuilt_cpp():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP accumulated batch training with bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_accumulate_batch_DDP_with_bucket_rebuilt_cpp"
    )
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"],
    level_mark="level1",
    card_mark="allcards")
def test_accumulate_batch_DDP_without_bucket_rebuilt_cpp():
    """
    Feature: Distributed Data Parallel(DDP).
    Description: DDP accumulated batch training without bucket rebuild in params executed order
    Expectation: Run success
    """
    ret = os.system("sysctl -w net.ipv4.ip_local_reserved_ports=30000-30015")
    ret = os.system(
        f"msrun --worker_num=4 --local_worker_num=4 --log_dir=msrun_log --join=True --master_port=8129\
            pytest -s distributed_data_parallel.py::test_accumulate_batch_DDP_without_bucket_rebuilt_cpp"
    )
    assert ret == 0
