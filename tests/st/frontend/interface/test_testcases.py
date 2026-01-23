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
"""Test interface."""
import pytest
from tests.mark_utils import arg_mark
from tests.st.frontend.utils.model_train_base import set_mode
from .cell.test_cell_load_state_dict import test_load_state_dict_and_hook_same
from .cell.test_cell_state_dict import test_state_dict_with_one_cell
from .container.test_sequentialcell import test_sequentialcell_input_list_conv2d_bn_relu
from .data_format.test_apply_momentum_3d_format import test_momentum_forward_input_3x8x4x12x32_lr_0001_momentum_00
from .data_format.test_sgd_3d_format import test_sgd_3d_forward_input_3x8x4x12x32_lr_01_momentum_00_epoch_2



@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
@pytest.mark.parametrize('mode', ["pynative", "KBK", "GE"])
def test_interface_testcases(mode):
    """
    Feature: test interface: cell, container and data format.
    Description: test interface: cell, container and data format.
    Expectation: the result match with expected result.
    """
    set_mode(mode)
    test_load_state_dict_and_hook_same()
    test_state_dict_with_one_cell()
    test_sequentialcell_input_list_conv2d_bn_relu()
    test_momentum_forward_input_3x8x4x12x32_lr_0001_momentum_00()
    test_sgd_3d_forward_input_3x8x4x12x32_lr_01_momentum_00_epoch_2()
