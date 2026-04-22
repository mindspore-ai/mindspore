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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards')
def test_train_step_pause():
    """
    Feature: train step pause.
    Description: When train step end, stop and check weather need switch nic.
    Expectation: Print the log, and sleep 1s.
    """
    return_code = os.system(
        "export MS_ENABLE_TFT='TSP:1';"
        "msrun --worker_num=4 --local_worker_num=4 --master_addr=127.0.0.1 --master_port=10741 --join=True "
        "train_step_pause.py --device_target=Ascend --dataset_path=/home/workspace/mindspore_dataset/mnist"
    )
    assert return_code == 0

    with open('worker_0.log', 'r') as f:
        lines = f.readlines()
        if all("sync_func is (" not in line for line in lines):
            raise ValueError("Set stream sync func failed!")
        if all("In tft pause train, current step is" not in line for line in lines):
            raise ValueError("Train step pause failed!")
