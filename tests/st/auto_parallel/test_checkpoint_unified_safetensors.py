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
"""Test runner for unified safetensors checkpoint saving with pipeline parallelism and shared parameters.

Tests checkpoint transformation and consolidation with safetensors format using msrun launcher.
"""
import os
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_unified_safetensors_pp_shared_param():
    '''
    Feature: safetensors pp_shared_param.
    Description: Test safetensors pp_shared_param.
    Expectation: Run success.
    '''
    ret = os.system("msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10809 "
                    "--join=True --log_dir=./test_checkpoint_unified pytest -s -v "
                    "checkpoint_unified_safetensors.py::test_unified_safetensors_pp_shared_param")
    assert ret == 0
