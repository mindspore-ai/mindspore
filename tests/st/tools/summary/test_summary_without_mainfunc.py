# Copyright 2023 Huawei Technologies Co., Ltd
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
import sys
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards",
          essential_mark="essential")
def test_summarycollector():
    """
    Feature: Test SummaryCollector in distribute training.
    Description: Run Summary script on 8 cards ascend computor, init() is not in main function.
    Expectation: No error occur.
    """
    if sys.platform != 'linux':
        return
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 "
                    "--master_addr=127.0.0.1 --master_port=10802 "
                    "--join=True --log_dir=./test_summary_without_mainfunc/msrun_log pytest -s "
                    "summary_net.py::test_summary_net")
    assert ret == 0
