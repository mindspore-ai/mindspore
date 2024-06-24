# Copyright 2022 Huawei Technologies Co., Ltd
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
from ge_test_utils import run_testcase
from tests.mark_utils import arg_mark


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_ge_graph_mode_with_jit_level():
    """
    Description: Graph Mode jit_level with GE.
    Expectation: Run by ge_device_context when jit_level.
    """
    run_testcase("ge_graph_mode_jit_level")
