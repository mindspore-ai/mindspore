# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
test feature of binding numa: msrun --bind_numa
"""
import re
import os
import pytest
import subprocess
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_true():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with auto bind numa policy.
    Expectation: Expected log in stdout.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_true.log"

    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa=True {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)

    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command:(?!.*numactl).*?", output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind (\d+) --cpunodebind (\d+) .*",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind (\d+) --cpunodebind (\d+) .*",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_false():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with False.
    Expectation: Disable bind numa and no expected log in stdout.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_false.log"

    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa=False {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)

    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command:(?!.*numactl).*?", output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command:(?!.*numactl).*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command:(?!.*numactl).*?",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_manual_only_worker():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy.
    Expectation: Bind numa for workers, expected log in stdout.
    """
    bind_numa_arg = '{"device0":["0-1"], "device1":["2-3"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_only_worker.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa='{bind_numa_arg}' {script}"
    )

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_numa arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command:(?!.*taskset -c).*?", output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind 0-1 --cpunodebind 0-1 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind 2-3 --cpunodebind 2-3 .*?",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_manual_single_worker():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy.
    Expectation: Bind numa for single worker, expected log in stdout.
    """
    bind_numa_arg = '{"device0":["0,  1", "2- 3"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_single_worker.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_numa='{bind_numa_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_numa arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command:(?!.*numactl).*?", output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind 0,1,2-3 --cpunodebind 0,1,2-3 .*?",
        output_log,
    )
    assert "Cannot find process[1] in args '--bind_numa'." in output_log
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command:(?!.*numactl).*?",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_manual_only_worker_wrong_sequence():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy.
    Expectation: Disable bind numa for workers and expected log in stdout.
    """
    bind_numa_arg = '{"device1":["0"], "device0":["1"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_only_worker_wrong_sequence.log"
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_numa='{bind_numa_arg}' {script}"

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_numa arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(r"Start scheduler process.*? Execute command:(?!.*numactl).*?", output_log)
    assert "Cannot find physical_device_id[0] for process[0] in args '--bind_numa'." in output_log
    assert re.search(r"Start worker process with rank id:0.*? Execute command:(?!.*numactl).*?", output_log)
    assert "Cannot find physical_device_id[1] for process[1] in args '--bind_numa'." in output_log
    assert re.search(r"Start worker process with rank id:1.*? Execute command:(?!.*numactl).*?", output_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_manual_only_scheduler():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy.
    Expectation: Bind numa for scheduler, expected log in stdout.
    """
    bind_numa_arg = '{"scheduler":["0"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_only_scheduler.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa='{bind_numa_arg}' {script}"
    )

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_numa arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command: numactl --membind 0 --cpunodebind 0 .*?",
        output_log,
    )
    assert "Cannot find process[0] in args '--bind_numa'." in output_log
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command:(?!.*numactl).*?",
        output_log,
    )
    assert "Cannot find process[1] in args '--bind_numa'." in output_log
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command:(?!.*numactl).*?",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards')
def test_msrun_bind_numa_manual_scheduler_and_worker():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy.
    Expectation: Bind numa for scheduler and workers, expected log in stdout.
    """
    bind_numa_arg = '{"device0":["0"], "scheduler":["2"], "device1":["1"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_scheduler_and_worker.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa='{bind_numa_arg}' {script}"
    )

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_numa arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command: numactl --membind 2 --cpunodebind 2 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind 0 --cpunodebind 0 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind 1 --cpunodebind 1 .*?",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_manual_empty_list():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy.
    Expectation: Bind numa for scheduler and workers, expected log in stdout.
    """
    bind_numa_arg = '{"device0":[], "scheduler":["2"], "device1":["0-1", "3"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_empty_list.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa='{bind_numa_arg}' {script}"
    )

    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")
    assert os.path.exists(output)

    # check results of msrun --bind_numa arg.
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command: numactl --membind 2 --cpunodebind 2", output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command:(?!.*numactl).*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind 0-1,3 --cpunodebind 0-1,3 .*?",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards')
def test_msrun_bind_numa_manual_wrong_key_value():
    """
    Feature: Runtime msrun --bind_numa arg.
    Description: Test msrun --bind_numa with customized bind numa policy and wrong parameters.
    Expectation: Wrong parameters make msrun failed.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)

    # Test for wrong value ["0~9"].
    bind_numa_arg = '{"device0":["0~1"], "scheduler":[], "device1":["2-3"]}'
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa='{bind_numa_arg}' {script}"
    )
    with pytest.raises(subprocess.CalledProcessError) as err_info:
        subprocess.run(
            f"{env} {msrun_cmd}", shell=True, check=True, capture_output=True, text=True
        )
    print(err_info.value.stderr, flush=True)
    assert (
        "NUMA 0~1 in ['0~1'] should be in format 'numaidX', 'numaidX,numaidY' or 'numaidX-numaidY'."
        in err_info.value.stderr
    )

    # Test for exceed NUMA limit ["300-500"].
    bind_numa_arg = '{"device0":["0-1"], "scheduler":[], "device1":["50-60"]}'
    msrun_cmd = "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "\
                f"--bind_numa='{bind_numa_arg}' {script}"
    with pytest.raises(subprocess.CalledProcessError) as err_info:
        subprocess.run(
            f"{env} {msrun_cmd}",
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
    print(err_info.value.stderr, flush=True)
    assert "NUMA id 50 in '--bind_numa' is not available." in err_info.value.stderr


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards')
def test_msrun_bind_numa_true_bind_core_true():
    """
    Feature: Runtime msrun --bind_numa and --bind_core arg.
    Description: Test msrun --bind_numa and --bind_core with auto bind numa policy.
    Expectation: Expected log in stdout.
    """
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_true_bind_core_true.log"

    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa=True --bind_core=True {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)

    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command:(?!.*numactl) (?!.*taskset).*?", output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind (\d+) --physcpubind (\d+)-(\d+) .*",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind (\d+) --physcpubind (\d+)-(\d+) .*",
        output_log,
    )


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards')
def test_msrun_bind_numa_manual_bind_core_manual():
    """
    Feature: Runtime msrun --bind_numa and --bind_core arg.
    Description: Test msrun --bind_numa and --bind_core with customized bind core policy.
    Expectation: Expected log in stdout.
    """
    bind_numa_arg = '{"device0":["0"], "scheduler":["2"], "device1":["1"]}'
    bind_core_arg = '{"scheduler":["48-71"], "device0":["0-23"], "device1":["24-47"]}'
    env = "export DISTRIBUTED=1;"
    env += "export GLOG_v=2;"

    real_path = os.path.realpath(os.getcwd())
    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = real_path + "/test_msrun_bind_numa_manual_bind_core_manual.log"

    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--bind_numa='{bind_numa_arg}' --bind_core='{bind_core_arg}' {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)

    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
        print(output_log, flush=True)
    assert return_code == 0
    assert re.search(
        r"Start scheduler process.*? Execute command: numactl --membind 2 --physcpubind 48-71 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind 0 --physcpubind 0-23 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind 1 --physcpubind 24-47 .*?",
        output_log,
    )
