
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
The tests of mindspore, used to test communication for cpu.
"""
import copy
import os
import signal
import subprocess
import time
from tests.mark_utils import arg_mark
@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_hccl_mint_cpu_ops():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case without async mccl init
    Expectation: success and worker log contains sync start log
    """
    log_dir = "test_cpu_sync_mccl_init"
    return_code = os.system(
        "export GLOG_v=2; msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 "
        f"--join=True --log_dir={log_dir} pytest -s test_comm_cpu.py"
    )
    assert return_code == 0
    result = subprocess.getoutput(
        f"grep -rn 'The worker node start sync start' ./{log_dir}/worker_0.log"
    )
    assert result.find("The worker node start sync start") != -1


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_hccl_mint_cpu_ops_async():
    """
    Feature: mpi run 8P case with async mccl init
    Description: test async mccl initialization by setting MS_DEV_RUNTIME_CONF
    Expectation: success and worker log contains async start log
    """
    log_dir = "test_cpu_async_mccl_init"
    return_code = os.system(
        "export GLOG_v=2; export MS_DEV_RUNTIME_CONF=async_mccl_init_comm:true; "
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 "
        f"--join=True --log_dir={log_dir} pytest -s test_comm_cpu.py"
    )
    assert return_code == 0
    result = subprocess.getoutput(
        f"grep -rn 'The worker node start async start' ./{log_dir}/worker_0.log"
    )
    assert result.find("The worker node start async start") != -1


def _create_worker_process(worker_id, cmd_list, log_dir, base_env, env):
    """
    Feature: test create worker process
    """
    log_file = f'{log_dir}/worker_{worker_id}.log'
    custom_env = base_env.copy()
    custom_env.update(env)
    custom_env['RANK_ID'] = str(worker_id)

    try:
        with open(log_file, 'w', encoding='utf-8') as file_handle:
            return subprocess.Popen(
                cmd_list,
                stdout=file_handle,
                stderr=subprocess.STDOUT,
                env=custom_env,
                text=True,
                start_new_session=True
            )
    except PermissionError:
        print(f"Error: No permission to write log file {log_file}")
        return None
    except (OSError, subprocess.SubprocessError) as e:
        print(f"Error: starting worker {worker_id}: {e}")
        return None


def _check_processes_status(processes, return_codes):
    """
    Feature: test check processes status
    """
    all_finished = True
    has_failed = False
    for i, p in enumerate(processes):
        if return_codes[i] is None:
            ret_code = p.poll()
            if ret_code is not None:
                return_codes[i] = ret_code
                print(f"Worker process: {i} has exited, ret_code: {ret_code}.")
                if ret_code != 0:
                    has_failed = True
            else:
                all_finished = False
    return all_finished, has_failed


def _handle_failed_workers(processes):
    """
    Feature: test handle failed workers
    """
    print("Some worker processes failed. Waiting 5 seconds before terminating all workers...")
    time.sleep(5)

    for i, p in enumerate(processes):
        if p.poll() is None:
            print(f"Terminating worker {i}...")
            p.terminate()

    for i, p in enumerate(processes):
        if p.poll() is None:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Worker {i} did not terminate, force killing...")
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)

    print("All worker processes have been terminated.")
    return 1


def run_workers(cmd_list=None, world_size=8, log_dir='output', env=None):
    """
    Feature: test run_workers
    """
    print(f"\nStart to run {world_size} worker processes, cmd_list: {cmd_list}, log_dir: {log_dir}, "
          f"input_env: {env}")

    if not cmd_list:
        print("Error: No command to execute!")
        return 1

    try:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: creating log directory {log_dir}: {e}")
        return 1

    env = env or {}
    if not isinstance(env, dict):
        print("Error: env must be a dictionary")
        return 1

    processes = []
    base_env = copy.deepcopy(os.environ)

    for i in range(world_size):
        worker_process = _create_worker_process(i, cmd_list, log_dir, base_env, env)
        if not worker_process:
            for p in processes:
                p.terminate()
            return 1
        processes.append(worker_process)
        print(f"Run worker process {i}, rank_id: {i}, log_dir: {log_dir}/worker_{i}.log.")

    try:
        return_codes = [None] * len(processes)
        all_finished = False
        has_failed = False

        while not all_finished and not has_failed:
            all_finished, has_failed = _check_processes_status(processes, return_codes)
            if not all_finished and not has_failed:
                time.sleep(0.1)

        if has_failed:
            return _handle_failed_workers(processes)

        print("All worker processes finished successfully.")
        return 0

    except KeyboardInterrupt:
        print("\nReceived interrupt, terminating all workers...")
        for p in processes:
            p.terminate()
        return 1


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_hccl_mint_init_with_init_method():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    cmd_list = ['pytest', '-sv', 'test_comm_cpu.py']
    world_size = 8
    log_dir = './init_with_init_method'
    env = {
        "INIT_APPROACH": "INIT_METHOD"
    }
    return_code = run_workers(cmd_list, world_size, log_dir, env)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_hccl_mint_init_with_tcpstore():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    cmd_list = ['pytest', '-sv', 'test_comm_cpu.py']
    world_size = 8
    log_dir = './init_with_tcpstore'
    env = {
        "INIT_APPROACH": "TCPSTORE"
    }
    return_code = run_workers(cmd_list, world_size, log_dir, env)
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_hccl_mint_cpu_ops1():
    """
    Feature: mpi run 8P case
    Description: mpi run 8P case
    Expectation: success
    """
    return_code = os.system(
        r"cp  test_comm_cpu.py test_comm_cpu1.py && "\
        r"sed -i 's/mindspore\.mint\.distributed\.distributed/mindspore.ops.communication/g' "\
        r"test_comm_cpu1.py && msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 "\
        r"--master_port=10666 --join=True pytest -s test_comm_cpu1.py"
    )
    assert return_code == 0


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_cpu_comm_op_timeout_env_priority():
    """
    Feature: MS_CPU_COMM_OP_TIMEOUT takes priority over MS_NODE_TIMEOUT.
    Description: When both MS_CPU_COMM_OP_TIMEOUT and MS_NODE_TIMEOUT are set,
                 comm_op_timeout_ should be read from MS_CPU_COMM_OP_TIMEOUT.
    Expectation: worker log contains the value set in MS_CPU_COMM_OP_TIMEOUT.
    """
    log_dir = "test_cpu_comm_op_timeout_env_priority"
    return_code = os.system(
        "export GLOG_v=1; export MS_CPU_COMM_OP_TIMEOUT=123; export MS_NODE_TIMEOUT=456; "
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 "
        f"--join=True --log_dir={log_dir} pytest -s test_comm_cpu.py::test_cpu_broadcast"
    )
    assert return_code == 0
    result = subprocess.getoutput(
        f"grep -rn 'cpu comm op exec timeout bave been set' ./{log_dir}/worker_0.log"
    )
    assert result.find("cpu comm op exec timeout bave been set 123 seconds") != -1


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards")
def test_cpu_comm_op_timeout_env_fallback():
    """
    Feature: fallback to MS_NODE_TIMEOUT when MS_CPU_COMM_OP_TIMEOUT is unset.
    Description: When MS_CPU_COMM_OP_TIMEOUT is not set but MS_NODE_TIMEOUT is,
                 comm_op_timeout_ should be read from MS_NODE_TIMEOUT.
    Expectation: worker log contains the value set in MS_NODE_TIMEOUT.
    """
    log_dir = "test_cpu_comm_op_timeout_env_fallback"
    return_code = os.system(
        "unset MS_CPU_COMM_OP_TIMEOUT; export GLOG_v=1; export MS_NODE_TIMEOUT=789; "
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=10666 "
        f"--join=True --log_dir={log_dir} pytest -s test_comm_cpu.py::test_cpu_broadcast"
    )
    assert return_code == 0
    result = subprocess.getoutput(
        f"grep -rn 'cpu comm op exec timeout bave been set' ./{log_dir}/worker_0.log"
    )
    assert result.find("cpu comm op exec timeout bave been set 789 seconds") != -1
