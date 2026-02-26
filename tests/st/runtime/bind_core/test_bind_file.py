# Copyright 2026 Huawei Technologies Co., Ltd
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
import json
from tests.mark_utils import arg_mark


real_path = os.path.realpath(os.getcwd())


def _generate_bind_file_json(output_path, config):
    """Generate a JSON configuration file for CPU and memory binding"""
    # Check if the file exists and delete it
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            print(f"Existing file deleted: {output_path}")
        except Exception as e:
            print(f"Warning: Failed to delete existing file {output_path}: {str(e)}")
            raise

    try:
        file_dir = os.path.dirname(output_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create directory for file {output_path}: {str(e)}")
        raise

    # Write configuration to JSON file
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"JSON configuration file generated successfully: {output_path}")
    except Exception as e:
        print(f"Failed to generate JSON file: {str(e)}")
        raise


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_0():
    """
    Feature: Pass bind_file to msrun --bind_numa and mindspore.runtime.set_cpu_affinity.
    Description: bind_cpu_mode=cpu, bind_memory_mode=numa; process binds main via numactl --physcpubind
                 and memory via --membind; modules bind via set_cpu_affinity.
    Expectation: Both process-level numactl and thread-level set_cpu_affinity policies take effect.
    """
    case_name = "test_bind_file_0"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa", "actor_thread_fix_bind": True},
        "bind_cpu": {
            "scheduler": {"main": "40-45"},
            "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
            "device1": {"main": "20-24", "runtime": "25-29", "pynative": "30-34", "minddata": "35-39"},
        },
        "bind_memory": {"device0": 0, "device1": 1, "scheduler": 2},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1; "
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"
    env += "export GLOG_v=1;"
    env += "export MS_DEV_RUNTIME_CONF='actor_thread_fix_bind:false';"

    script = real_path + "/run_bind.py"
    assert os.path.exists(script)
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12333 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    # Process-level binding (numactl).
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert re.search(
        r"Start scheduler process.*? Execute command: numactl --membind 2 --physcpubind 40-45 .*?",
        output_log
    )
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind 0 --physcpubind 0-4 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind 1 --physcpubind 20-24 .*?",
        output_log,
    )

    # Thread-level binding (set_cpu_affinity).
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated: {'runtime': [5, 6, 7, 8, 9], 'pynative': [10, 11, 12, 13, 14], " \
           "'minddata': [15, 16, 17, 18, 19]}" in worker_0_log
    assert "Module bind core policy generated: {'runtime': [25, 26, 27, 28, 29], 'pynative': [30, 31, 32, 33, 34], " \
           "'minddata': [35, 36, 37, 38, 39]}" in worker_1_log
    assert re.search(r"This module: 0 is assigned a bind core list: .*?\{\}", worker_0_log)
    assert re.search(r"This module: 1 is assigned a bind core list: .*?\{5, 6, 7, 8, 9\}", worker_0_log)
    assert re.search(r"Success to bind core to 5 for thread \d+", worker_0_log)
    assert re.search(r"Success to bind core to 25 for thread \d+", worker_1_log)
    assert re.search(r"This module: 2 is assigned a bind core list: .*?\{10, 11, 12, 13, 14\}", worker_0_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_1():
    """
    Feature: Pass bind_file to msrun --bind_numa only (bind_cpu_mode=numa, bind_memory_mode=numa).
    Description: Bind CPU/NUMA by node, no set_cpu_affinity.
    Expectation: numactl --membind/--cpunodebind are applied per process, no taskset.
    """
    case_name = "test_bind_file_1"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "numa"},
        "bind_cpu": {"device0": 0, "device1": 1},
        "bind_memory": {"device0": 0, "device1": 1, "scheduler": "2-3,4"},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12334 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert re.search(r"Start scheduler process.*? Execute command: numactl --membind 2-3,4 .*?", output_log)
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --membind 0 --cpunodebind 0 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --membind 1 --cpunodebind 1 .*?",
        output_log,
    )
    # set_cpu_affinity should not be invoked (no module policy log).
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated" not in worker_0_log
    assert "Module bind core policy generated" not in worker_1_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_2():
    """
    Feature: Pass bind_file with cpu-mode binding cores and no memory binding.
    Description: main bound via taskset, modules bound via set_cpu_affinity.
    Expectation: taskset applied; set_cpu_affinity uses module mapping.
    """
    case_name = "test_bind_file_2"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none", "actor_thread_fix_bind": False},
        "bind_cpu": {
            "scheduler": {"main": "40-45"},
            "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
            "device1": {"main": "20-24", "runtime": "25-29", "pynative": "30-34", "minddata": "35-39"},
        },
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"
    env += "export GLOG_v=1;"
    env += "export MS_DEV_RUNTIME_CONF='actor_thread_fix_bind:true';"

    script = real_path + "/run_bind.py"
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12335 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert re.search(r"Start scheduler process.*? Execute command: taskset -c 40-45 .*?", output_log)
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c 0-4 .*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c 20-24 .*?", output_log)

    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated: {'runtime': [5, 6, 7, 8, 9], 'pynative': [10, 11, 12, 13, 14], " \
           "'minddata': [15, 16, 17, 18, 19]}" in worker_0_log
    assert "Module bind core policy generated: {'runtime': [25, 26, 27, 28, 29], 'pynative': [30, 31, 32, 33, 34], " \
           "'minddata': [35, 36, 37, 38, 39]}" in worker_1_log
    assert re.search(r"This module: 0 is assigned a bind core list: .*?\{\}", worker_0_log)
    assert re.search(r"This module: 1 is assigned a bind core list: .*?\{5, 6, 7, 8, 9\}", worker_0_log)
    assert re.search(r"Success to bind core to .*?\{5, 6, 7, 8, 9\} for thread \d+", worker_0_log)
    assert re.search(r"This module: 2 is assigned a bind core list: .*?\{10, 11, 12, 13, 14\}", worker_0_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_3():
    """
    Feature: Pass bind_file with cpu-mode binding cores only for main.
    Description: Only process-level taskset is applied; no set_cpu_affinity expected.
    Expectation: taskset applied per process; no module binding log.
    """
    case_name = "test_bind_file_3"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
        "bind_cpu": {
            "device0": {"main": "0-4"},
            "device1": {"main": "20-24"},
        },
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12336 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert re.search(r"Start worker process with rank id:0.*? Execute command: taskset -c 0-4 .*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: taskset -c 20-24 .*?", output_log)
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated: {}" in worker_0_log
    assert "set_cpu_affinity is not enabled because the environment does not meet the basic "\
           "conditions for binding core." in worker_0_log
    assert "Module bind core policy generated: {}" in worker_1_log
    assert "set_cpu_affinity is not enabled because the environment does not meet the basic "\
           "conditions for binding core." in worker_1_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_4():
    """
    Feature: Pass bind_file with numa CPU binding only.
    Description: Bind CPU by numa node; no set_cpu_affinity expected.
    Expectation: numactl --cpunodebind is used; no module binding log.
    """
    case_name = "test_bind_file_4"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "none"},
        "bind_cpu": {"device0": 0, "device1": 1},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12337 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert re.search(
        r"Start worker process with rank id:0.*? Execute command: numactl --cpunodebind 0 .*?",
        output_log,
    )
    assert re.search(
        r"Start worker process with rank id:1.*? Execute command: numactl --cpunodebind 1 .*?",
        output_log,
    )
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated" not in worker_0_log
    assert "Module bind core policy generated" not in worker_1_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_5():
    """
    Feature: Pass bind_file with memory-only binding.
    Description: bind_cpu_mode=none, bind_memory_mode=numa.
    Expectation: numactl --membind is used; no set_cpu_affinity.
    """
    case_name = "test_bind_file_5"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "none", "bind_memory_mode": "numa"},
        "bind_memory": {"device0": 0, "device1": 1},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12338 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert re.search(r"Start worker process with rank id:0.*? Execute command: numactl --membind 0 .*?", output_log)
    assert re.search(r"Start worker process with rank id:1.*? Execute command: numactl --membind 1 .*?", output_log)
    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated" not in worker_0_log
    assert "Module bind core policy generated" not in worker_1_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_bind_file_6():
    """
    Feature: Pass bind_file with cpu-mode modules only (no main).
    Description: Only set_cpu_affinity should bind threads; no taskset/numactl.
    Expectation: msrun launches without taskset/numactl prefix; set_cpu_affinity binds modules.
    """
    case_name = "test_bind_file_6"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none"},
        "bind_cpu": {
            "device0": {"runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
            "device1": {"runtime": "25-29", "pynative": "30-34", "minddata": "35-39"},
        },
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"
    env += "export GLOG_v=1;"

    script = real_path + "/run_bind.py"
    output = case_path + f"/{case_name}.log"
    worker_0 = f"{case_path}/worker_0.log"
    worker_1 = f"{case_path}/worker_1.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code == 0
    assert os.path.exists(output) and os.path.exists(worker_0) and os.path.exists(worker_1)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "taskset" not in output_log and "numactl" not in output_log

    with open(worker_0, "r", encoding="utf-8") as f:
        worker_0_log = f.read()
    with open(worker_1, "r", encoding="utf-8") as f:
        worker_1_log = f.read()
    assert "Module bind core policy generated: {'runtime': [5, 6, 7, 8, 9], 'pynative': [10, 11, 12, 13, 14], " \
           "'minddata': [15, 16, 17, 18, 19]}" in worker_0_log
    assert "Module bind core policy generated: {'runtime': [25, 26, 27, 28, 29], 'pynative': [30, 31, 32, 33, 34], " \
           "'minddata': [35, 36, 37, 38, 39]}" in worker_1_log
    assert re.search(r"This module: 1 is assigned a bind core list: .*?\{5, 6, 7, 8, 9\}", worker_0_log)
    assert re.search(r"Success to bind core to .*?\{5, 6, 7, 8, 9\} for thread \d+", worker_0_log)
    assert re.search(r"This module: 2 is assigned a bind core list: .*?\{10, 11, 12, 13, 14\}", worker_0_log)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_bind_file_not_exist():
    """
    Feature: Validate bind_numa json file path.
    Description: Pass a non-existent .json path to msrun --bind_numa.
    Expectation: msrun exits with file-not-exist error.
    """
    case_name = "test_except_bind_file_not_exist"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"./{case_name}.json"
    os.makedirs(case_path, exist_ok=True)

    script = real_path + "/run_bind.py"
    env = "export DISTRIBUTED=1;"
    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12340 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    return_code = os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert return_code != 0
    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert f"bind_numa json file '{bind_file_path}' does not exist." in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_invalid_bind_cpu():
    """
    Feature: Validate bind_cpu in bind file.
    Description: In bind_cpu, the type of value should be dict.
    Expectation: msrun/set_cpu_affinity exits with error for invalid bind_cpu.
    """
    case_name = "test_except_invalid_bind_cpu"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none"},
        "bind_cpu": {
            "scheduler": "40-45",
            "device0": "0-4",
            "device1": "20-24",
        },
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    script = real_path + "/run_bind.py"

    # Validate msrun
    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "taskset" not in output_log and "numactl" not in output_log
    assert "'bind_cpu_mode' is 'cpu', in 'bind_cpu', type of value for scheduler:40-45 "\
           "should be dict, but got <class 'str'>." in output_log

    # Validate set_cpu_affinity
    env = "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_set_cpu_affinity.log"
    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert f"bind_file '{bind_file_path}' entry for device0, type of 0-4 should be dict, "\
           "but got <class 'str'>. Skip enabling set_cpu_affinity." in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_invalid_bind_cpu_mode():
    """
    Feature: Validate bind_cpu_mode in bind file.
    Description: bind_cpu_mode only supports cpu/numa/none (case-insensitive).
    Expectation: msrun/set_cpu_affinity exits with error for invalid bind_cpu_mode.
    """
    case_name = "test_except_invalid_bind_cpu_mode"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "invalid", "bind_memory_mode": "none"},
        "bind_cpu": {"device0": {"main": "0-4"}},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    script = real_path + "/run_bind.py"

    # Validate msrun
    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "'bind_cpu_mode' only supports 'cpu'/'numa'/'none'" in output_log

    # Validate set_cpu_affinity
    env = "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_set_cpu_affinity.log"
    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "'bind_cpu_mode' only supports 'cpu'/'numa'/'none'" in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_invalid_bind_memory_mode():
    """
    Feature: Validate bind_memory_mode in bind file.
    Description: bind_memory_mode only supports numa/none (case-insensitive).
    Expectation: msrun/set_cpu_affinity exits with error for invalid bind_memory_mode.
    """
    case_name = "test_except_invalid_bind_memory_mode"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "invalid"},
        "bind_cpu": {"device0": {"main": "0-4"}},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    script = real_path + "/run_bind.py"

    # Validate msrun
    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "'bind_memory_mode' only supports 'numa'/'none'" in output_log

    # Validate set_cpu_affinity
    env = "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_set_cpu_affinity.log"
    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "'bind_memory_mode' only supports 'numa'/'none'" in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_invalid_actor_thread_fix_bind():
    """
    Feature: Validate actor_thread_fix_bind in bind file.
    Description: actor_thread_fix_bind only supports True/False (case-insensitive).
    Expectation: set_cpu_affinity exits with error for invalid actor_thread_fix_bind.
    """
    case_name = "test_except_invalid_actor_thread_fix_bind"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {
            "bind_cpu_mode": "cpu",
            "bind_memory_mode": "none",
            "actor_thread_fix_bind": "invalid",
        },
        "bind_cpu": {"device0": {"runtime": "5-9", "pynative": "10-14", "minddata": "15-19"}},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    script = real_path + "/run_bind.py"

    # Validate msrun
    env = "export DISTRIBUTED=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "'actor_thread_fix_bind' only supports bool type True/False" not in output_log

    # Validate set_cpu_affinity
    env = "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    output = case_path + f"/{case_name}_set_cpu_affinity.log"
    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "'actor_thread_fix_bind' only supports bool type True/False" in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_bind_file_hash_mismatch():
    """
    Feature: Validate bind file hash consistency between msrun and set_cpu_affinity.
    Description: Mutate bind file after msrun launch to trigger hash mismatch.
    Expectation: set_cpu_affinity reports hash mismatch and msrun exits with error.
    """
    case_name = "test_except_bind_file_hash_mismatch"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
        "bind_cpu": {
            "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
            "device1": {"main": "20-24", "runtime": "25-29", "pynative": "30-34", "minddata": "35-39"},
        },
        "bind_memory": {"device0": 0, "device1": 1},
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += "export EDIT_BIND_FILE=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert f"The '{bind_file_path}' content has changed since msrun was launched. " \
           "Please keep the file consistent." in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_exceed_cpu():
    """
    Feature: Validate available CPU in bind file.
    Description: Unavailable CPU in bind file raises error.
    Expectation: Unavailable CPU in bind file raises error.
    """
    case_name = "test_except_exceed_cpu"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "numa"},
        "bind_cpu": {
            "device0": {"main": "0-4"},
            "device1": {"main": "1000-1004"},
        }
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "CPU id 1000 in {\'main\': \'1000-1004\'} for device1 is not available." in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_except_exceed_numa():
    """
    Feature: Validate available NUMA in bind file.
    Description: Unavailable NUMA in bind file raises error.
    Expectation: Unavailable NUMA in bind file raises error.
    """
    case_name = "test_except_exceed_numa"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "numa", "bind_memory_mode": "numa"},
        "bind_cpu": {
            "device0": "0-1",
            "device1": "100-101",
        }
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export DISTRIBUTED=1;"
    env += "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    script = real_path + "/run_bind.py"

    output = case_path + f"/{case_name}_msrun.log"
    msrun_cmd = (
        "msrun --worker_num=2 --local_worker_num=2 --master_port=12339 --join=True "
        f"--log_dir={case_path} --bind_numa={bind_file_path} {script}"
    )
    os.system(f"{env} {msrun_cmd} > {output} 2>&1")

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "NUMA id 100 in 100-101 for device1 is not available." in output_log


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_main_when_no_msrun():
    """
    Feature: Pass only the "bind_file" to set_cpu_affinity; it holds the highest priority and binds the main thread.
    Description: set_cpu_affinity should include main from bind file when MSRUN_BIND_FILE is not set.
    Expectation: When the module bind policy log contains the keyword 'main', ensure proper usage of bind_file.
    """
    case_name = "test_main_when_no_msrun"
    case_path = real_path + f"/{case_name}"
    bind_file_path = f"{case_path}/{case_name}.json"
    bind_file_content = {
        "bind_config": {"bind_cpu_mode": "cpu", "bind_memory_mode": "none"},
        "bind_cpu": {
            "device0": {"main": "0-4", "runtime": "5-9", "pynative": "10-14", "minddata": "15-19"},
        },
    }
    _generate_bind_file_json(bind_file_path, bind_file_content)
    assert os.path.exists(bind_file_path)

    env = "export THREAD_BIND=1;"
    env += f"export BIND_FILE_PATH={bind_file_path};"

    affinity_cpu_list = '["0-10"]'
    module_to_cpu_dict = '{"main": [9, 10], "minddata": [7, 8], "runtime": [5, 6], "pynative": [3, 4]}'
    env += f"export AFFINITY_CPU_LIST='{affinity_cpu_list}';"
    env += f"export MODULE_TO_CPU_DICT='{module_to_cpu_dict}';"

    script = real_path + "/run_bind.py"

    output = case_path + f"/{case_name}.log"
    cmd = f"{env} python {script} > {output} 2>&1"
    os.system(cmd)

    assert os.path.exists(output)
    with open(output, "r", encoding="utf-8") as f:
        output_log = f.read()
    assert "Module bind core policy generated: {'runtime': [5, 6, 7, 8, 9], 'pynative': [10, 11, 12, 13, 14], " \
           "'minddata': [15, 16, 17, 18, 19], 'main': [0, 1, 2, 3, 4]}" in output_log
