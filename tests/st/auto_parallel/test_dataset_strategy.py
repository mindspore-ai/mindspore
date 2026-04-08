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
def test_msrun_data_strategy_dataset_parallel():
    '''
    Feature: AutoParallel(cell).dataset_strategy(config)
    Description: Test interface AutoParallel(cell).dataset_strategy("data_parallel")
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10915 "
                    "--join=True --log_dir=./dataset_strategy_logs/data_parallel pytest -s -v "
                    "dataset_strategy.py::test_dataset_strategy_data_parallel")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_data_strategy_fullbatch():
    '''
    Feature: AutoParallel(cell).dataset_strategy(config)
    Description: Test interface AutoParallel(cell).dataset_strategy("full_batch")
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10916 "
                    "--join=True --log_dir=./dataset_strategy_logs/full_batch pytest -s -v "
                    "dataset_strategy.py::test_dataset_strategy_full_batch")
    assert ret == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_dataset_strategy_2_1_1_2_1():
    '''
    Feature: AutoParallel(cell).dataset_strategy(config)
    Description: Test interface AutoParallel(cell).dataset_strategy(((2,1,1),(2,1)))
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10917 "
                    "--join=True --log_dir=./dataset_strategy_logs/2_1_1_2_1 pytest -s -v "
                    "dataset_strategy.py::test_dataset_strategy_using_tuple")
    assert ret == 0
