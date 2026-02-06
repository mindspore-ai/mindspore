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
"""
Test case for arf recovery and hccl status record
"""

import os
import json
import shutil
from pathlib import Path
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_msrun_comm_subgraph_8p():
    """
    Feature: test hcom create/destroy interface.
    Description: test graceful exit, save ckpt after exit training process.
    Expectation: none.
    """
    self_path = os.path.split(os.path.realpath(__file__))[0]
    return_code = os.system(f"bash {self_path}/bash_run.sh")
    assert return_code == 0

    logs = os.path.join(os.getcwd(), "logs")
    if os.path.exists(logs):
        shutil.rmtree(logs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_save_hccl_op_status_8p():
    """
    Feature: test save hccl op exe status.
    Description: test save hccl op exe status.
    Expectation: none.
    """

    self_path = os.path.split(os.path.realpath(__file__))[0]
    save_path = os.path.join(self_path, "save_json")
    fp = Path(save_path)
    fp.mkdir(parents=True, exist_ok=True)
    context = {"HCCL_STATUS_SAVE_PATH": str(save_path),
               "HCCL_STATUS_SAVE_INTERVAL": 10}

    with open(f"{self_path}/ccae_config.json", "w", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=4)

    return_code = os.system(f"bash {self_path}/run_save_hccl_status.sh")
    assert return_code == 0

    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    logs = os.path.join(os.getcwd(), "logs")
    if os.path.exists(logs):
        shutil.rmtree(logs)
