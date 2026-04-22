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
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_auto_parallel_sharding_propagation():
    '''
    Feature: Auto parallel, strategy:((1,1),(1,2)), parallel_mode is sharding_paopagation
    Description: Test auto parallel mode.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10918 "
                    "--join=True --log_dir=./auto_parallel_logs/sharding_paopagation pytest -s -v "
                    "auto_parallel_train.py::test_auto_parallel_sharding_propagation")
    assert ret == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_msrun_auto_parallel_recursive_programming():
    '''
    Feature: Auto parallel, parallel_mode is recursive_programming
    Description: Test auto parallel mode.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10919 "
                    "--join=True --log_dir=./auto_parallel_logs/recursive_programming pytest -s -v "
                    "auto_parallel_train.py::test_auto_parallel_recursive_programming")
    assert ret == 0
