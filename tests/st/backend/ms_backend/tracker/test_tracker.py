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
Test module for tracker.
"""
import os
import csv
from tests.mark_utils import arg_mark


def run_simple_tracker_command(cmd, log_path, tracker_ir_pash, memory_block_csv_path):
    if os.path.isfile(log_path):
        os.remove(log_path)
    if os.path.isfile(tracker_ir_pash):
        os.remove(tracker_ir_pash)
    if os.path.isfile(memory_block_csv_path):
        os.remove(memory_block_csv_path)
    os.system(cmd)
    # no tracker_graph.ir
    assert not os.path.exists(tracker_ir_pash)
    # Check if memory_block.csv exists
    assert os.path.exists(memory_block_csv_path), "memory_block.csv file does not exist"
    # no user_tasks
    with open(memory_block_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            assert row['user_tasks'] == '', f"Expected empty user_tasks but got {row['user_tasks']}"
    # Check if log file exists
    assert os.path.exists(log_path), "Log file does not exist"
    # Check if log contains "Simple tracker, skip dump"
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "ERROR" not in log_content, "Log contains 'ERROR' message"
        assert "Simple tracker, skip dump" in log_content, "Log does not contain 'Simple tracker, skip dump' message"
    if os.path.isfile(log_path):
        os.remove(log_path)
    if os.path.isfile(tracker_ir_pash):
        os.remove(tracker_ir_pash)
    if os.path.isfile(memory_block_csv_path):
        os.remove(memory_block_csv_path)


def run_tracker_command(cmd, log_path, tracker_ir_pash, memory_block_csv_path):
    if os.path.isfile(log_path):
        os.remove(log_path)
    if os.path.isfile(tracker_ir_pash):
        os.remove(tracker_ir_pash)
    if os.path.isfile(memory_block_csv_path):
        os.remove(memory_block_csv_path)
    os.system(cmd)
    # tracker_graph.ir
    assert os.path.exists(tracker_ir_pash)
    # Check if memory_block.csv exists
    assert os.path.exists(memory_block_csv_path), "memory_block.csv file does not exist"
    # user_tasks
    with open(memory_block_csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        has_non_empty_tasks = False
        for row in reader:
            if row['user_tasks'] != '':
                has_non_empty_tasks = True
                break
        assert has_non_empty_tasks, "Expected at least one row with non-empty user_tasks but found none"
    # Check if log file exists
    assert os.path.exists(log_path), "Log file does not exist"
    # Check if log contains "Dump graph to file" message
    with open(log_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
        assert "ERROR" not in log_content, "Log contains 'ERROR' message"
        assert "Dump graph to file" in log_content, "Log does not contain 'Dump graph to file' message"
    if os.path.isfile(log_path):
        os.remove(log_path)
    if os.path.isfile(tracker_ir_pash):
        os.remove(tracker_ir_pash)
    if os.path.isfile(memory_block_csv_path):
        os.remove(memory_block_csv_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='dryrun_only', essential_mark='essential')
def test_simple_tracker():
    """
    Feature: simple tracker
    Description: simple tracker ok
    Expectation: run success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_simple_tracker_command(f"bash {sh_path}/dryrun_simple_tracker.sh", f"{sh_path}/simple_tracker.log",
                               f"{sh_path}/rank_1/tracker_graph.ir", f"{sh_path}/rank_1/memory_block.csv")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='dryrun_only', essential_mark='essential')
def test_tracker():
    """
    Feature: tracker
    Description: tracker ok
    Expectation: run success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_tracker_command(f"bash {sh_path}/dryrun_tracker.sh tracker_pynative.py tracker1.log", f"{sh_path}/tracker1.log",
                        f"{sh_path}/rank_0/tracker_graph.ir", f"{sh_path}/rank_0/memory_block.csv")

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='dryrun_only', essential_mark='essential')
def test_resize_storage_tracker():
    """
    Feature: tracker
    Description: tracker ok
    Expectation: run success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_tracker_command(f"bash {sh_path}/dryrun_tracker.sh resize_storage_tracker.py tracker2.log",
                        f"{sh_path}/tracker2.log", f"{sh_path}/tracker_graph.ir",
                        f"{sh_path}/memory_block.csv")
