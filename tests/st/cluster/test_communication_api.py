# Copyright 2026 Huawei Technologies Co., Ltd
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
This test is for mindspore.communication API.
"""
import pytest

import mindspore as ms
from mindspore.communication import init, create_group, get_rank

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})

def test_create_group_exception():
    """
    Feature: test exception scenario for create_group.
    Description: create_group with sub-group in all ranks.
    Expectation: exception.
    """
    init()
    sub_group_name = "sub_group_0_7"
    sub_group_rank_list = [0, 7]
    if get_rank() in sub_group_rank_list:
        create_group(sub_group_name, sub_group_rank_list)
    else:
        with pytest.raises(RuntimeError) as error_info:
            create_group(sub_group_name, sub_group_rank_list)
        assert "Group sub_group_0_7 doesn't contain the global rank" in str(error_info.value)


def test_get_rank_exception():
    """
    Feature: test exception scenario for get_rank.
    Description: get_rank with sub-group in all ranks.
    Expectation: exception.
    """
    init()
    sub_group_name = "sub_group_0_7"
    sub_group_rank_list = [0, 7]
    if get_rank() in sub_group_rank_list:
        create_group(sub_group_name, sub_group_rank_list)
        get_rank(sub_group_name)
    else:
        with pytest.raises(RuntimeError) as error_info:
            get_rank(sub_group_name)
        assert "The group sub_group_0_7 does not exist." in str(error_info.value)
