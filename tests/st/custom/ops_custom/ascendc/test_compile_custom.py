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
"""Testing custom operator's offline compilation"""
from tests.mark_utils import arg_mark
import tempfile
from compile_utils import compile_custom_run


@arg_mark(plat_marks=['platform_ascend'], level_mark='level4', card_mark='onecard', essential_mark='unessential')
def test_custom_compile():
    """
    Feature: Custom op testcase
    Description: test case for offline compilation"
    Expectation: generate custom run
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        compile_custom_run(temp_dir)
