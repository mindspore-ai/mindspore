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
"""Test model evaluate."""
import pytest
from .test_evaluate import test_evaluate_input_3
from .test_evaluate_network import test_eval_network_net1_net2
from tests.st.frontend.utils.model_train_base import set_mode
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ["pynative", "KBK", "GE"])
def test_all_evaluate_testcases(mode):
    """
    Feature: test interface: cell, container and data format.
    Description: test interface: cell, container and data format.
    Expectation: the result match with expected result.
    """
    set_mode(mode)
    test_eval_network_net1_net2()
    test_evaluate_input_3()
