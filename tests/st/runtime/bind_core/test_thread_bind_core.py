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
test feature of binding core: mindspore.runtime.set_cpu_affinity with msrun --bind_core
"""
import re
import os
import pytest
import subprocess
import mindspore as ms
from tests.mark_utils import arg_mark


def _check_env_valid_cpu_resource():
    """
    Check if the program can correctly recognize the CPU resources on the environment.
    """
    try:
        result = subprocess.run(
            ["cat", "/sys/fs/cgroup/cpuset/cpuset.cpus"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(f"Available CPU range is {result}.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Fail to execute command 'cat /sys/fs/cgroup/cpuset/cpuset.cpus', because {e}")
        return False


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_bind_core_auto():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api which automatically bind thread core.
    Expectation: Core bound for module and threads.
    """
    if not _check_env_valid_cpu_resource():
        print("Skip this ST, as the environment is not suitable for thread bind core.")
        return

    env = "export THREAD_BIND=1;"
    env += "export GLOG_v=1;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    output = real_path + "/thread_bind_core_auto.log"
    assert os.path.exists(script)

    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert re.search(r"Module bind core policy generated: \{'main': \[([\d, ]*)\]\}", output_log)
    assert re.search(r"This module: 0 is assigned a bind core list: .+?", output_log)
    assert re.search("Skip to bind thread core for 'pynative'", output_log)
    assert re.search("Skip to bind thread core for 'runtime actor'", output_log)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0',
          card_mark='onecard', essential_mark='essential', mem_peak=4.00)
def test_bind_core_manual():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api which manually bind thread core for dynamic shape.
    Expectation: Core bound for module and threads as input affinity_cpu_list and module_to_cpu_dict.
    """
    if not _check_env_valid_cpu_resource():
        print("Skip this ST, as the environment is not suitable for thread bind core.")
        return

    affinity_cpu_list = '["0-10", "21-30"]'
    module_to_cpu_dict = '{"main": [0, 1, 2, 3], "minddata": [4, 5], "other": [6, 7], \
                         "runtime": [8, 9], "pynative": [10, 11, 21, 100]}'

    env = "export THREAD_BIND=1;"
    env += "export GLOG_v=1;"
    env += f"export AFFINITY_CPU_LIST='{affinity_cpu_list}';"
    env += f"export MODULE_TO_CPU_DICT='{module_to_cpu_dict}';"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/thread_bind_core_manual.log"

    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    manual_policy_str = ("Module bind core policy generated: {'main': [0, 1, 2, 3], "
                         "'minddata': [4, 5], 'runtime': [8, 9], 'pynative': [10, 21]}")
    assert manual_policy_str in output_log
    assert re.search(r"This module: 0 is assigned a bind core list: .*?\{0, 1, 2, 3\}", output_log)
    assert re.search(r"This module: 1 is assigned a bind core list: .*?\{8, 9\}", output_log)
    assert re.search(r"Success to bind core to .*?\{8, 9\} for thread \d+", output_log)
    assert re.search(r"This module: 2 is assigned a bind core list: .*?\{10, 21\}", output_log)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_bind_core_empty_module_assigned():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity empty module_to_cpu_dict.
    Expectation: Expected log in stdout.
    """
    if not _check_env_valid_cpu_resource():
        print("Skip this ST, as the environment is not suitable for thread bind core.")
        return

    env = "export THREAD_BIND=1;"
    env += "export GLOG_v=2;"
    env += "export MODULE_TO_CPU_DICT='{}';"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/thread_bind_core_empty_module.log"

    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert "Module bind core policy generated: {}" in output_log


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_bind_core_manual_no_available_cpus():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api manually input unavailable cpu list.
    Expectation: RuntimeError reported.
    """
    affinity_cpu_list = ["300-500"]
    with pytest.raises(RuntimeError) as err_info:
        ms.runtime.set_cpu_affinity(True, affinity_cpu_list)
    assert f"set in affinity_cpu_list:{affinity_cpu_list} is not available." in str(err_info)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
def test_bind_core_repeatly_call():
    """
    Feature: Runtime set_cpu_affinity api.
    Description: Test runtime.set_cpu_affinity api repeatedly called..
    Expectation: RuntimeError reported.
    """
    affinity_cpu_list = ["0-10"]
    ms.runtime.set_cpu_affinity(True, affinity_cpu_list)
    with pytest.raises(RuntimeError) as err_info:
        ms.runtime.set_cpu_affinity(False)
    assert "The 'mindspore.runtime.set_cpu_affinity' cannot be set repeatedly." in str(err_info)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_msrun_bind_true_thread_bind_auto():
    """
    Feature: Runtime set_cpu_affinity api with msrun --bind_core arg.
    Description: Test runtime.set_cpu_affinity and msrun --bind_core with auto bind core policy.
    Expectation: Expected log in stdout.
    """
    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += "export GLOG_v=1;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_true_thread_bind_auto.log"
    msrun_path = real_path + "/msrun_bind_true_thread_bind_auto"
    worker_0 = f'{msrun_path}/worker_0.log'

    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--log_dir={msrun_path} --bind_core=True {script}"
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    assert os.path.exists(worker_0)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Execute command: taskset -c (\d+)-(\d+) .*", output_log)

    # check results of mindspore.runtime.set_cpu_affinity API.
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    assert re.search(r"Module bind core policy from msrun: \{'main': \[([\d, ]*)\]\}", worker_0_log)
    assert re.search(r"This module: 0 is assigned a bind core list: .+?", worker_0_log)
    assert re.search("Skip to bind thread core for 'pynative'", worker_0_log)
    assert re.search("Skip to bind thread core for 'runtime actor'", worker_0_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_msrun_bind_manual_thread_bind_manual():
    """
    Feature: Runtime set_cpu_affinity api with msrun --bind_core arg.
    Description: Test runtime.set_cpu_affinity and msrun --bind_core with customized bind core policy.
    Expectation: Expected log in stdout.
    """
    bind_core_arg = '{"scheduler":["16-19"], "device0":["11-15"], "device1":["31-35"]}'
    affinity_cpu_list = '["0-10"]'
    affinity_cpu_list_2 = '["20-30"]'
    module_to_cpu_dict = '{"main": [0, 1, 2, 3], "minddata": [4, 5], "other": [6], \
                         "runtime": [7, 8], "pynative": [9, 10, 11, 21, 100]}'

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += "export GLOG_v=1;"
    env += f"export AFFINITY_CPU_LIST='{affinity_cpu_list}';"
    env += f"export AFFINITY_CPU_LIST_2='{affinity_cpu_list_2}';"
    env += f"export MODULE_TO_CPU_DICT='{module_to_cpu_dict}';"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_thread_bind_manual.log"
    msrun_path = real_path + "/msrun_bind_manual_thread_bind_manual"
    worker_0 = f'{msrun_path}/worker_0.log'
    worker_1 = f'{msrun_path}/worker_1.log'
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--log_dir={msrun_path} --bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)
    assert os.path.exists(worker_0)
    assert os.path.exists(worker_1)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command: taskset -c 16-19 .*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c 11-15 .*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c 31-35 .*?", output_log)

    # check results of mindspore.runtime.set_cpu_affinity API.
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    manual_policy_str = ("Module bind core policy generated: {'main': [0, 1, 2, 3], "
                         "'minddata': [4, 5], 'runtime': [7, 8], 'pynative': [9, 10]}")
    assert manual_policy_str in worker_0_log
    assert re.search(r"This module: 0 is assigned a bind core list: .*?\{0, 1, 2, 3\}", worker_0_log)
    assert re.search(r"This module: 1 is assigned a bind core list: .*?\{7, 8\}", worker_0_log)
    assert re.search(r"Success to bind core to .*?\{7, 8\} for thread \d+", worker_0_log)
    assert re.search(r"This module: 2 is assigned a bind core list: .*?\{9, 10\}", worker_0_log)
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    manual_policy_str = ("Module bind core policy generated: {'main': [20, 21, 22, 23], "
                         "'minddata': [24, 25], 'runtime': [27, 28], 'pynative': [29, 30]}")
    assert manual_policy_str in worker_1_log
