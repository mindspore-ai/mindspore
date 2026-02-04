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
"""Test dynamic_lr."""
import pytest
from tests.mark_utils import arg_mark
from tests.st.frontend.utils.model_train_base import set_mode
from .test_adam_optim import test_adam_true_1_1_0
from .test_adamax_optim import test_adamax_group_dynamic
from .test_adamw_optim import test_adamw_true_1_1_0
from .test_constant_lr import test_constant_lr_epoch_0_step_1_no_grouping
from .test_cosine_annealing_lr import test_cosine_annealing_lr_epoch_0_step_1_no_grouping
from .test_cosine_annealing_warm_restarts import test_cosine_annealing_warm_restarts_epoch_1_step_0_group_0
from .test_nadam_optim import test_nadam_group_dynamic
from .test_radam_optim import test_radam_group_dynamic


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.01)
@pytest.mark.parametrize('mode', ["pynative", "KBK", "GE"])
def test_dynamic_lr(mode):
    """
    Feature: test dynamic lr.
    Description: test dynamic lr.
    Expectation: the result match with expected result.
    """
    set_mode(mode)
    test_adam_true_1_1_0()
    test_adamax_group_dynamic()
    test_adamw_true_1_1_0()
    test_constant_lr_epoch_0_step_1_no_grouping()
    test_cosine_annealing_lr_epoch_0_step_1_no_grouping()
    test_cosine_annealing_warm_restarts_epoch_1_step_0_group_0()
    test_nadam_group_dynamic()
    test_radam_group_dynamic()
