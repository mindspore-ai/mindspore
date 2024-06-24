# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_ops_flash_attention_score_tnd():
    """
    Feature: Test the accuracy of TND query when cutting out.
    Description: Test function flash attention score forward.
    Expectation: The result of TND standalone and semi_parallel is equal.
    """
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 " \
        "--master_port=10969 --join=True --log_dir=./fa_logs " \
        "pytest -s flash_attention_score.py::test_flash_attention_score_tnd"
    )
    assert return_code == 0
