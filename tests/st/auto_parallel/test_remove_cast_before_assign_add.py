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
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="allcards", essential_mark="unessential")
def test_msrun_remove_cast_before_assign_add():
    """
    Feature: remove_cast_before_assign_add run semi_auto_parallel
    Description: Test remove_cast_before_assign_add feature.
    Expectation: Run success.
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret = os.system(f"source {sh_path}/env.sh && msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 "
                    "--master_port=10969 --join=True "
                    "--log_dir=./remove_cast_before_assign_add_logs pytest -s -v remove_cast_before_assign_add.py")
    assert ret == 0
