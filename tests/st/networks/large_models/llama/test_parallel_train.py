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
"""
Test module for testing the paralleled llama interface used for mindformers.
How to run this:
pytest tests/st/test_model/test_llama_model/test_parallel_train.py
"""
import os
from multiprocessing.pool import Pool


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
        os.system(f"cat {results[idx][1]}")
    assert error_idx == []


def test_train():
    """
    Feature: Trainer.train()
    Description: Test context parallel trainer for train.
    Expectation: AssertionError
    """
    ascend_home_path = os.getenv('ASCEND_HOME_PATH')
    if not ascend_home_path:
        os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"

    sh_path = os.path.split(os.path.realpath(__file__))[0]
    commands = [(f"export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 && "
                 f"bash {sh_path}/msrun_launch_llama.sh 4 test_train 8128",
                 f"{sh_path}/test_train/worker_0.log"),
                (f"export ASCEND_RT_VISIBLE_DEVICES=4,5 && "
                 f"bash {sh_path}/msrun_launch_llama.sh 2 test_train_cp 8129",
                 f"{sh_path}/test_train_cp/worker_0.log"),
                (f"export ASCEND_RT_VISIBLE_DEVICES=6,7 && "
                 f"bash {sh_path}/msrun_launch_llama.sh 2 test_train_dp 8131",
                 f"{sh_path}/test_train_dp/worker_0.log")
                ]

    with Pool(len(commands)) as pool:
        results = list(pool.imap(run_command, commands))
    check_results(commands, results)
