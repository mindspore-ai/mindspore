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
"""testcase for one-sided communication based on symmetric memory"""

import os
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_oneside_gather_8p():
    """
    Feature: one-sided Gather communication based on symmetric memory.
    Description: Verify Gather communication in symmetric memory scenario with 8 cards.
    Expectation: rank-0 output tensor should successfully receive data sent by producer process.
    """
    return_code = os.system("msrun --worker_num=8 --local_worker_num=8 --join=True pytest -s test_gather.py")
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_oneside_gather_4p():
    """
    Feature: one-sided Gather communication based on symmetric memory.
    Description: Verify Gather communication in symmetric memory scenario with 4 cards.
    Expectation: rank-0 output tensor should successfully receive data sent by producer process.
    """
    return_code = os.system("msrun --worker_num=4 --local_worker_num=4 --join=True pytest -s test_gather.py")
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_oneside_pull_graph():
    """
    Feature: one-sided Pull communication based on symmetric memory.
    Description: Verify Pull communication in symmetric memory scenario with 2 cards.
    Expectation: rank-1 output tensor should successfully get data from rank-0.
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True pytest -s test_pull_graph.py")
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_oneside_push_graph():
    """
    Feature: one-sided Push communication based on symmetric memory.
    Description: Verify Push communication in symmetric memory scenario with 2 cards.
    Expectation: rank-0 should successfully put data to rank-1.
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True pytest -s test_push_graph.py")
    assert return_code == 0

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
@test_utils.run_test_with_On
def test_oneside_compare():
    """
    Feature: one-sided signal set and wait based on symmetric memory.
    Description: Verify signal set and wait in symmetric memory scenario with 2 cards.
    Expectation: rank-1 should successfully wait for the signal set by rank-0.
    """
    return_code = os.system("msrun --worker_num=2 --local_worker_num=2 --join=True pytest -s test_compare.py")
    assert return_code == 0
