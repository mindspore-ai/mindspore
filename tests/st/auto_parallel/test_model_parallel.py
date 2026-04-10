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

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_model_programming():
    """
    Feature: AutoParallel(cell) in model parallel dimension
    Description: Train in Model.train way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10911 "
                    "--join=True --log_dir=./model_parallel_logs/model_programming_1_1_1_2 pytest -s -v "
                    "model_parallel.py::test_model_parallel_model_programming")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_model_parallel_functional_programming():
    """
    Feature: AutoParallel(cell) in model parallel dimension
    Description: Train in functional programming way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10912 "
                    "--join=True --log_dir=./model_parallel_logs/functional_programming_1_1_1_2 pytest -s -v "
                    "model_parallel.py::test_model_parallel_functional_programming")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_data_parallel_model_programming():
    """
    Feature: AutoParallel(cell) in data parallel dimension
    Description: Train in Model.train way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10913 "
                    "--join=True --log_dir=./model_parallel_logs/model_programming_8_1_1_1 pytest -s -v "
                    "model_parallel.py::test_data_parallel_model_programming")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_data_parallel_functional_programming():
    """
    Feature: AutoParallel(cell) in data parallel dimension
    Description: Train in functional programming way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10914 "
                    "--join=True --log_dir=./model_parallel_logs/functional_programming_8_1_1_1 pytest -s -v "
                    "model_parallel.py::test_data_parallel_functional_programming")
    assert ret == 0
