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
"""Test checkpoint."""
import pytest
from tests.mark_utils import arg_mark
from tests.st.frontend.utils.model_train_base import set_mode
from .callback.test_callback import test_callback_basic_6_insertion_point_check
from .callback.test_callback_advanced import test_lambdacallback_train_metrics, test_lambdacallback_eval_metrics
from .test_checkpoint_append import test_load_save_checkpoint_append_dic_string_enc
from .test_checkpoint_async_save import test_checkpoint_async_save_set_append_info_str_in_list
from .test_checkpoint_exception_save import test_checkpoint_exception_save_true_train_error_new_path
from .test_checkpoint_policy_check import test_checkpoint_only_time_strategy_more_traintime_seconds160_minutes2_step0_max0_false
from .test_checkpoint_specify_prefix import test_load_checkpoint_async_base
from .test_initial_epoch import test_epoch_is_5_exception_save_true_initial_epoch_is_3_train
from .test_model_checkpoint_callback import test_model_checkpoint_prefix_directory_callable


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4756.75)
@pytest.mark.parametrize('mode', ["pynative", "KBK", "GE"])
def test_checkpoint_testcases(mode):
    """
    Feature: test checkpoint.
    Description: test checkpoint.
    Expectation: the result match with expected result.
    """
    set_mode(mode)
    test_callback_basic_6_insertion_point_check()
    test_lambdacallback_train_metrics()
    test_lambdacallback_eval_metrics()
    test_load_save_checkpoint_append_dic_string_enc()
    test_checkpoint_async_save_set_append_info_str_in_list()
    test_checkpoint_exception_save_true_train_error_new_path()
    test_checkpoint_only_time_strategy_more_traintime_seconds160_minutes2_step0_max0_false()
    test_load_checkpoint_async_base()
    test_epoch_is_5_exception_save_true_initial_epoch_is_3_train()
    test_model_checkpoint_prefix_directory_callable()
