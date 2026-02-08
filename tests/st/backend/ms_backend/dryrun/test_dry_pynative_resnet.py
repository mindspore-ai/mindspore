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
import re
import shutil
from tests.mark_utils import arg_mark

def check_has_vmm_log(log_file_path):
    with open(log_file_path, 'r') as file:
        for line in file:
            if 'VMM is enabled' in line:
                return
    assert False, "No VMM log found in log file: " + log_file_path

def check_log_for_error(log_file_path):
    with open(log_file_path, 'r') as file:
        for line in file:
            if '[ERROR]' in line:
                assert False, "Error found in log file: " + log_file_path
    print("No error found in log file: ", log_file_path)


def get_used_max_memory(log_file_path):
    used_peak = 0
    with open(log_file_path, 'r') as file:
        for line in file:
            print(f"log debug: {log_file_path}, {line}", flush=True)
            if "Used peak memory" in line:
                match = re.search(r'\d+', line)
                if match:
                    used_peak = max(used_peak, int(match.group()))
    return used_peak


def check_file_exists_and_not_empty(file_path):
    if not os.path.exists(file_path):
        assert False, f"File '{file_path}' does not exist."
    if os.path.getsize(file_path) == 0:
        assert False, f"File '{file_path}' exists but is empty."
    print(f"File '{file_path}' exists and is not empty.")


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='allcards',
          essential_mark='essential')
def test_dry_pynative_resnet50_ascend_8p():
    """
    Feature: PyNative ResNet50 8P
    Description: test PyNative ResNet50 8p with mpirun
    Expectation: success, return_code==0
    """
    if os.path.exists("real_run"):
        shutil.rmtree("real_run")
    if os.path.exists("dry_run"):
        shutil.rmtree("dry_run")
    # real run
    os.system("GLOG_v=1 MS_ALLOC_CONF=\"acl_allocator:False,memory_tracker:True\" "\
              "msrun --worker_num=8 --local_worker_num=8 "\
              "--master_addr=127.0.0.1  --master_port=10969 --join=True "\
              "python test_pynative_resnet.py >stdout.log 2>&1")
    os.system("mkdir real_run")
    os.system("mv rank* real_run")
    os.system("mv *.log real_run")
    # dryrun
    os.system("GLOG_v=1 MS_ALLOC_CONF=\"acl_allocator:False,memory_tracker:True\"  MS_SIMULATION_LEVEL=1 "\
              "msrun --worker_num=8 --local_worker_num=8 --sim_level=1 "\
              "--master_addr=127.0.0.1  --master_port=10969 --join=True "\
              "python test_pynative_resnet.py >stdout.log 2>&1")
    os.system("mkdir dry_run")
    os.system("mv rank* dry_run")
    os.system("mv *.log dry_run")
    # compare
    check_log_for_error("real_run/worker_0.log")
    check_log_for_error("dry_run/worker_0.log")
    check_has_vmm_log("real_run/worker_0.log")
    dry_run_memory = get_used_max_memory("dry_run/worker_0.log")
    real_run_memory = get_used_max_memory("real_run/worker_0.log")
    check_file_exists_and_not_empty("dry_run/rank_0/tracker_graph.ir")
    check_file_exists_and_not_empty("real_run/rank_0/tracker_graph.ir")
    print("dry_run_memory: ", dry_run_memory)
    print("real_run_memory: ", real_run_memory)
    assert dry_run_memory == real_run_memory
    if os.path.exists("real_run"):
        shutil.rmtree("real_run")
    if os.path.exists("dry_run"):
        shutil.rmtree("dry_run")
