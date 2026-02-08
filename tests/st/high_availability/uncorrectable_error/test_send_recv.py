# Copyright 2024 Huawei Technologies Co., Ltd
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
Test senv_recv interface
"""
import os

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_send_recv():
    """
    Feature: Test send_recv interface.
    Description: Test send_recv interface.
    Expectation: The interface send_recv can work successfully.
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"bash {sh_path}/run_send_recv.sh {sh_path}/send_recv.py")
    assert ret == 0
