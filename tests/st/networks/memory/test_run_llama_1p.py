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
Test module for parallel training of Llama models using Mindformers at jit_level O2.
"""
import os
import subprocess
import csv
import re
from tests.mark_utils import arg_mark


def run_command(cmd, log_path, tracker_path, somas_check, enable_somas):
    memory_csv_filename = tracker_path + "/memory_block.csv"
    task_csv_filename = tracker_path + "/task.csv"
    if os.path.isfile(memory_csv_filename):
        os.remove(memory_csv_filename)
    if os.path.isfile(task_csv_filename):
        os.remove(task_csv_filename)
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    for line in open(log_path):
        print(line, end='', flush=True)

    eager_free_para = "Eager free count :"
    eager_free_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (eager_free_para, log_path)],
        shell=True)
    eager_free_cnt = str(eager_free_output, 'utf-8').strip()
    if enable_somas:
        assert int(eager_free_cnt) <= 2
    else:
        assert int(eager_free_cnt) == 0

    somas_para = "allocate somas merged blocks"
    somas_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (somas_para, log_path)],
        shell=True)
    somas_cnt = str(somas_output, 'utf-8').strip()
    assert somas_cnt == str(somas_check)

    print(int(eager_free_cnt), int(somas_cnt), flush=True)

    max_memory_size = 0
    used_peak = 0
    actual_peak_memory = 0
    with open(log_path, 'r') as file:
        for line in file:
            if "MindSpore Used memory size" in line:
                match = re.search(r'\d+', line)
                if match:
                    max_memory_size = max(max_memory_size, int(match.group()))
            if "Used peak memory" in line:
                match = re.search(r'\d+', line)
                if match:
                    used_peak = max(used_peak, int(match.group()))
            if "Actual peak memory" in line:
                match = re.search(r'\d+', line)
                if match:
                    actual_peak_memory = max(actual_peak_memory, int(match.group()))

    assert max_memory_size >= used_peak
    assert actual_peak_memory >= used_peak

    print(actual_peak_memory * 1024 * 1024, flush=True)

    with open(memory_csv_filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            csv_actual_peak_memory = row['actual_peak_memory']
            assert int(csv_actual_peak_memory) <= actual_peak_memory * 1024 * 1024

    if os.path.isfile(memory_csv_filename):
        os.remove(memory_csv_filename)
    if os.path.isfile(task_csv_filename):
        os.remove(task_csv_filename)
    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_somas():
    """
    Feature: Trainer.train()
    Description: Test context parallel trainer for train.
    Expectation: AssertionError
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"export MS_ALLOC_CONF='memory_tracker_path:{sh_path}/somas'&&bash {sh_path}/run_1p.sh somas",
                f"{sh_path}/somas.log", f"{sh_path}/somas", 10, True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
def test_no_somas():
    """
    Feature: Trainer.train()
    Description: Test context parallel trainer for train.
    Expectation: AssertionError
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"export MS_ALLOC_CONF='memory_tracker_path:{sh_path}/no_somas'&&bash {sh_path}/run_1p.sh no_somas",
                f"{sh_path}/no_somas.log", f"{sh_path}/no_somas", 0, False)
