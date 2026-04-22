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

import os
import subprocess
from mindspore import Tensor
from tests.mark_utils import arg_mark

def msrun_cross_cluster(rank_size=8):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # run test
    result = subprocess.getoutput(
        f"msrun --worker_num={rank_size} --local_worker_num={rank_size} --master_addr=127.0.0.1 --master_port=8118 "\
        f"--log_dir=ms_run --join=True --cluster_time_out=600 run_move_to.py"
    )
    test_passed = "run success, the loss is" in result

    if not test_passed:
        return False, "test move_to failed, please check the log for more details."
    return True, "test move_to success."

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_move_to_2_cards():
    '''
    Feature: test move_to in 2 cards
    Description: Test MoveTo in 2 cards scenarios.
    Expectation: Run success, the net can train.
    '''
    result, msg = msrun_cross_cluster(rank_size=2)
    assert result, msg


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards")
def test_tensor_move_to():
    '''
    Feature: Test Tensor move_to
    Description: Test Tensor move_to
    Expectation: Run success
    '''
    x = Tensor([1, 2, 3]).move_to("Ascend")
    assert "Ascend" in x.device

    assert "CPU" in x.move_to("CPU").device
