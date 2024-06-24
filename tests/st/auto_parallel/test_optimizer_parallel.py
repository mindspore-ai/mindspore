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


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_msrun_sit_optimizer_parallel():
    '''
    Feature: Optimizer parallel.
    Description: Test optimizer parallel feature along with model parallel.
    Expectation: Run success.
    '''
    ret = os.system("export GLOG_v=2 && msrun --worker_num=4 --local_worker_num=4 "
                    "--master_addr=127.0.0.1 --master_port=10969 "
                    "--join=True --log_dir=./sit_optimizer_parallel_logs pytest -s -v "
                    "optimizer_parallel.py::test_optimizer_parallel_auto_4p_6_parameter_same_strategy_1_1_2_1_momentum")
    assert ret == 0
