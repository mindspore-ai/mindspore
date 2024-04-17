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
import pytest
from tests.security_utils import security_off_wrap


@pytest.mark.level0
@pytest.mark.env_single
@security_off_wrap
def test_summarycollector():
    """
    Feature: Test SummaryCollector in distribute trainning.
    Description: Run Summary script on 8 cards ascend computor, init() is not in main function.
    Expectation: No error occur.
    """
    if sys.platform != 'linux':
        return
    exec_network_cmd = 'cd {0}; bash run_summary_without_mainfunc.sh'.format(
        os.path.split(os.path.realpath(__file__))[0])
    ret = os.system(exec_network_cmd)
    assert not ret
