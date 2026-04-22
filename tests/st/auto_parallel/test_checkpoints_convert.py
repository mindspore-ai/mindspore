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
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_checkpoints_convert_by_layout():
    """
    Feature: Test checkpoints convert with layout.
    Description: Test distributed checkpoints convert specified by layout.
    Expectation: The convert checkpoints is correct.
    """
    os.system("rm -rf ./test_checkpoints_convert_by_layout/")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 " \
        "--master_port=10805 --join=True " \
        "--log_dir=./test_checkpoints_convert_by_layout/msrun_log " \
        "pytest -s checkpoints_convert.py::test_checkpoints_convert_by_layout"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_checkpoints_convert_by_layout_with_opt_shard_safetensor():
    """
    Feature: Test checkpoints convert with layout and opt shard safetensor.
    Description: Test distributed checkpoints convert specified by layout.
    Expectation: The convert checkpoints is correct.
    """
    os.system("rm -rf ./test_checkpoints_convert_by_layout_with_opt_shard_size_2_safetensor/")
    return_code = os.system(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 " \
        "--master_port=10805 --join=True " \
        "--log_dir=./test_checkpoints_convert_by_layout_with_opt_shard_size_2_safetensor/msrun_log " \
        "pytest -s checkpoints_convert.py::test_checkpoints_convert_by_layout_with_opt_shard_safetensor"
    )
    assert return_code == 0
