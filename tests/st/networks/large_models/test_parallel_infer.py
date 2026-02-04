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
Test the paralleled infer interface used for mindformers.
"""
import os

from multiprocessing.pool import Pool
from tests.mark_utils import arg_mark

cur_dir = os.path.dirname(os.path.abspath(__file__))


def run_command(command_info):
    cmd, log_path = command_info
    ret = os.system(cmd)
    return ret, log_path


def check_results(commands, results):
    error_idx = [_ for _ in range(len(results)) if results[_][0] != 0]
    for idx in error_idx:
        print(f"testcase {commands[idx]} failed. please check log {results[idx][1]}.")
        os.system(f"grep -E 'ERROR|error|Error' {results[idx][1]} -C 5")
    assert error_idx == []


class TestInferParallel:
    """A test class for testing pipeline."""

    @staticmethod
    def setup_method():
        ascend_home_path = os.getenv('ASCEND_HOME_PATH')
        if not ascend_home_path:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/cann"

    @arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
    def test_base_cases(self):
        """
        Feature: Infer interface
        Description: Test parallel interface for training and prediction.
        Expectation: AssertionError
        """
        commands = [
            (f"export ASCEND_RT_VISIBLE_DEVICES=0,1 && export LCAL_COMM_ID=127.0.0.1:10068 && msrun --worker_num=2 "
             f"--local_worker_num=2 --master_port=8222 --log_dir=parallel_qwen2_0_5b_predict_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_qwen2_0_5b_predict_mp2",
             'parallel_qwen2_0_5b_predict_mp2/worker_0.log'),  # command, log_path
            (f"export ASCEND_RT_VISIBLE_DEVICES=2,3 && export LCAL_COMM_ID=127.0.0.1:10070 && msrun --worker_num=2 "
             f"--local_worker_num=2 --master_port=8230 --log_dir=parallel_deepseek_r1_bf16_predict_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_deepseek_r1_bf16_predict_mp2",
             'parallel_deepseek_r1_bf16_predict_mp2/worker_0.log'),
            (f"export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 && export LCAL_COMM_ID=127.0.0.1:10074 && msrun --worker_num=4 "
             f"--local_worker_num=4 --master_port=8240 --log_dir=parallel_qwen2_0_5b_predict_dp2_mp2 --join=True "
             f"{cur_dir}/run_parallel.py --mode parallel_qwen2_0_5b_predict_dp2_mp2",
             'parallel_qwen2_0_5b_predict_dp2_mp2/worker_0.log'),
        ]

        with Pool(len(commands)) as pool:
            results = list(pool.imap(run_command, commands))
        check_results(commands, results)
