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
test feature of binding core: msrun --bind_core
"""
import re
import os
import pytest
import subprocess
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_true():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with auto bind core policy.
    Expectation: Expected log in stdout.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_true.log"

    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core=True {script}"
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)

    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c (\d+)-(\d+) .*", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c (\d+)-(\d+) .*", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_false():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with False.
    Expectation: Disable bind core and no expected log in stdout.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_false.log"

    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core=False {script}"
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)

    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command:(?!.*taskset -c).*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_manual_only_worker():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy.
    Expectation: Bind core for workers, expected log in stdout.
    """
    bind_core_arg = '{"device0":["10-19"], "device1":["20-29"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_only_worker.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c 10-19 .*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c 20-29 .*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_manual_single_worker():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy.
    Expectation: Bind core for single worker, expected log in stdout.
    """
    bind_core_arg = '{"device0":["10-19"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_single_worker.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c 10-19 .*?", output_log)
    assert "Cannot find process[1] in args '--bind_core'." in output_log
    assert re.search(r"Start worker process with rank id:1.*? Execute command:(?!.*taskset -c).*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_manual_only_worker_wrong_sequence():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy.
    Expectation: Disable bind core for workers and expected log in stdout.
    """
    bind_core_arg = '{"device1":["10-19"], "device0":["20-29"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_only_worker_wrong_sequence.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert "Cannot find physical_device_id[0] for process[0] in args '--bind_core'." in output_log
    assert re.search(r"Start worker process with rank id:0.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert "Cannot find physical_device_id[1] for process[1] in args '--bind_core'." in output_log
    assert re.search(r"Start worker process with rank id:1.*? Execute command:(?!.*taskset -c).*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_manual_only_scheduler():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy.
    Expectation: Bind core for scheduler, expected log in stdout.
    """
    bind_core_arg = '{"scheduler":["0-9"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_only_scheduler.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command: taskset -c 0-9 .*?", output_log)
    assert "Cannot find process[0] in args '--bind_core'." in output_log
    assert re.search(r"Start worker process with rank id:0.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert "Cannot find process[1] in args '--bind_core'." in output_log
    assert re.search(r"Start worker process with rank id:1.*? Execute command:(?!.*taskset -c).*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_manual_scheduler_and_worker():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy.
    Expectation: Bind core for scheduler and workers, expected log in stdout.
    """
    bind_core_arg = '{"device0":["10-19"], "scheduler":["0-9"], "device1":["20-29"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_scheduler_and_worker.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command: taskset -c 0-9 .*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c 10-19 .*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c 20-29 .*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards',
        essential_mark='essential')
def test_msrun_bind_manual_empty_list_and_single_core():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy.
    Expectation: Bind core for scheduler and workers, expected log in stdout.
    """
    bind_core_arg = '{"device0":[], "scheduler":["10"], "device1":["20-29", "31"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)
    output = real_path + "/msrun_bind_manual_empty_list_and_single_core.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_core arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command: taskset -c 10", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command:(?!.*taskset -c).*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c 20-29,31 .*?", output_log)


def test_msrun_bind_manual_wrong_key_value():
    """
    Feature: Runtime msrun --bind_core arg.
    Description: Test msrun --bind_core with customized bind core policy and wrong parameters.
    Expectation: Wrong parameters make msrun failed.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind_core.py"
    assert os.path.exists(script)

    # Test for wrong key "DEVICE0".
    bind_core_arg = '{"DEVICE0":[], "scheduler":[], "device1":["20-29"]}'
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"
    with pytest.raises(subprocess.CalledProcessError) as err_info:
        subprocess.run(
            f"{env} {msrun_cmd}",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
    print(err_info.value.stderr, flush=True)
    assert "Key DEVICE0 must be in format 'scheduler' or 'deviceX' (X ≥ 0)" in err_info.value.stderr

    # Test for wrong value "0-10".
    bind_core_arg = '{"device0":"0-10", "scheduler":[], "device1":["20-29"]}'
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"
    with pytest.raises(subprocess.CalledProcessError) as err_info:
        subprocess.run(
            f"{env} {msrun_cmd}",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
    print(err_info.value.stderr, flush=True)
    assert "Value for device0:0-10 should be a list, but got <class 'str'>" in err_info.value.stderr

    # Test for wrong value [0-9].
    bind_core_arg = '{"device0":[["0-9"]], "scheduler":[], "device1":["20-29"]}'
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"
    with pytest.raises(subprocess.CalledProcessError) as err_info:
        subprocess.run(
            f"{env} {msrun_cmd}",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
    print(err_info.value.stderr, flush=True)
    assert "CPU ['0-9'] in [['0-9']] should be a string" in err_info.value.stderr

    # Test for wrong value ["0~9"].
    bind_core_arg = '{"device0":["0~9"], "scheduler":[], "device1":["20-29"]}'
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_core='{bind_core_arg}' {script}"
    with pytest.raises(subprocess.CalledProcessError) as err_info:
        subprocess.run(
            f"{env} {msrun_cmd}",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
    print(err_info.value.stderr, flush=True)
    assert "CPU 0~9 in ['0~9'] should be in format 'cpuidX' or 'cpuidX-cpuidY'" in err_info.value.stderr
