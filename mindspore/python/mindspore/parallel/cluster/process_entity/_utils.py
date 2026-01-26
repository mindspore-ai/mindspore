# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Utils for ms_run"""
import os
import json
import socket
import ipaddress
import subprocess
import mindspore.log as logger
from mindspore.runtime.thread_bind_core import (
    _get_cpu_info,
    _get_physical_device_id,
    _get_available_cpus,
    _get_available_numas,
    _auto_generate_bind_core_strategy,
    _auto_generate_bind_numa_strategy,
    _equal_distribute_cpu_strategy,
    _equal_distribute_numa_strategy,
)

CURRENT_IP = None


class BindContext:
    """
    Carry bind_numa/bind_core switches and related available resources.
    """
    def __init__(self):
        self.bind_numa = False
        self.bind_core = False
        self.available_numas = []
        self.available_cpus = []
        self.device_to_numa_map = {}
        self.device_to_cpu_map = {}


def _generate_cmd(cmd, cmd_args, local_rank, bind_context):
    """
    Generates a command string to execute a Python script in the background.
    """
    arg_bind_numa = bind_context.bind_numa
    arg_bind_core = bind_context.bind_core
    available_numas= bind_context.available_numas
    available_cpus = bind_context.available_cpus
    device_to_numa_map = bind_context.device_to_numa_map
    device_to_cpu_map = bind_context.device_to_cpu_map

    # scheduler process only supports to bind core/bind numa manually.
    if local_rank == -1:
        arg_bind_core = False if not isinstance(arg_bind_core, dict) else arg_bind_core
        arg_bind_numa = False if not isinstance(arg_bind_numa, dict) else arg_bind_numa

    # 1. Build the base command (e.g., ['python', 'my_script.py', '--arg', 'val'])
    base_cmd = _generate_cmd_args_list(cmd, cmd_args)

    # 2. Build the binding prefix dynamically
    binding_prefix = []
    # If NUMA binding is requested, it takes precedence for CPU binding as well.
    if arg_bind_numa:
        affinity_numa_str = _generate_bind_numa_strategy(
            local_rank,
            device_to_numa_map,
            arg_bind_numa,
            available_numas,
            available_cpus,
        )
        if affinity_numa_str is not None:
            # Prepare the numactl base command
            numa_cmd = ['numactl', '--membind', affinity_numa_str]

            # Check if core binding is also requested
            affinity_cpu_str = None
            if arg_bind_core:
                affinity_cpu_str = _generate_bind_core_strategy(local_rank, device_to_cpu_map, arg_bind_core)

            # If a CPU core string is available, use it with numactl's --physcpubind
            if affinity_cpu_str is not None:
                numa_cmd.extend(['--physcpubind', affinity_cpu_str])
            else:
                # If no specific core is given, bind to all cores on the NUMA node
                numa_cmd.extend(['--cpunodebind', affinity_numa_str])

            binding_prefix = numa_cmd
            # Since we've handled core binding via numactl, we don't need taskset anymore
            arg_bind_core = False 

    # If core binding is requested AND it wasn't handled by NUMA
    if arg_bind_core:
        affinity_cpu_str = _generate_bind_core_strategy(local_rank, device_to_cpu_map, arg_bind_core)
        if affinity_cpu_str is not None:
            binding_prefix = ['taskset', '-c', affinity_cpu_str]

    # 3. Combine the prefix and the base command
    final_cmd = binding_prefix + base_cmd

    return final_cmd


def _generate_cmd_args_list(cmd, cmd_args):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaultly use 'python' to launch the job.
        return ['python', cmd] + cmd_args
    return [cmd] + cmd_args


def _generate_url(addr, port):
    """
    Generates a url string by addr and port
    """
    url = ""
    return url


def _get_local_ip(ip_address):
    """
    Get current IP address.
    """
    global CURRENT_IP
    if CURRENT_IP is None:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect((ip_address, 0))
            CURRENT_IP = s.getsockname()[0]
            s.close()
        except Exception as e:
            raise RuntimeError("Get local ip has failed. Please verify that the accessible address has been "
                               "specified in the '--master_address' parameter") from e
    return CURRENT_IP


def _is_local_ip(ip_address):
    """
    Check if the current input IP address is a local IP address.
    """
    p = os.popen("ip -j addr")
    addr_info_str = p.read()
    p.close()
    current_ip = _get_local_ip(ip_address)
    if not addr_info_str:
        return current_ip == ip_address

    addr_infos = json.loads(addr_info_str)
    for info in addr_infos:
        for addr in info["addr_info"]:
            if addr["local"] == ip_address:
                logger.info(f"IP address found on this node. Address info:{addr}. Found address:{ip_address}")
                return True
    return False


def _convert_addr_to_ip(master_addr):
    """
    Check whether the input parameter 'master_addr' is IPv4. If a hostname is inserted, it will be converted
    to IP and then set as master host's IP.
    """
    try:
        ipaddress.IPv4Address(master_addr)
        return master_addr
    except ipaddress.AddressValueError:
        try:
            ip_address = socket.gethostbyname(master_addr)
            logger.info(f"Convert input host name:{master_addr} to ip address:{ip_address}.")
            return ip_address
        except socket.gaierror as e:
            raise RuntimeError("DNS resolution has failed. Please verify that the correct hostname has been "
                               "specified in the '--master_address' parameter") from e


def _send_scale_num(url, scale_num):
    """
    Send an HTTP request to a specified URL, informing scale_num.
    """
    return ""


def _check_command_availability(cmd):
    """
    Use `which` to check if command exists in PATH
    """
    target_cmd = cmd.strip()
    try:
        subprocess.run(
            ['which', target_cmd],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5
        )
        return True
    except subprocess.CalledProcessError:
        logger.warning(f"Command '{target_cmd}' is not found in system PATH.")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"Check for command '{target_cmd}' timed out.")
        return False
    except FileNotFoundError:
        logger.warning("'which' command not found in system.")
        return False
    except PermissionError:
        logger.warning("No permission to execute 'which' command.")
        return False
    except OSError as e:
        logger.warning(f"Unexpected error when checking command '{target_cmd}': {str(e)}")
        return False


def _validate_customized_bind_dict(bind_dict, available_set, label):
    """
    Validate the availability of CPU/NUMA node IDs from bind_dict.
    """
    for ranges in bind_dict.values():
        for seg in ",".join(ranges).split(","):
            if not seg:
                continue
            if "-" in seg:
                start, end = seg.split("-", 1)
                ids = range(int(start), int(end) + 1)
            else:
                ids = [int(seg)]
            for _id in ids:
                if _id not in available_set:
                    raise RuntimeError(f"{label} id {_id} in '--bind_{label.lower()}' is not available.")


def _generate_bind_context(local_worker_num, arg_bind_numa, arg_bind_core):
    """
    1. Validate the availability of mandatory system commands required for CPU core binding and memory binding
       in the current environment.
    2. Automatically generate the CPU/memory affinity policy when the auto-binding mode for CPU/memory is activated.
    3. Perform validity and format checks on the custom binding policy if it is provided as an input parameter.
    """
    available_numas = _get_available_numas()
    available_cpus = _get_available_cpus()
    device_to_numa_map = {}
    device_to_cpu_map = {}

    if arg_bind_core:
        if not _check_command_availability("taskset"):
            logger.warning("Command 'taskset' is not supported, arg '--bind_core' takes no effect!")
            arg_bind_core = False
        elif not available_cpus:
            logger.warning("Failed to acquire available cpu info from '/sys/fs/cgroup/cpuset/cpuset.cpus'. "
                           "Will not enable bind core feature.")
            arg_bind_core = False

    if isinstance(arg_bind_core, dict):
        _validate_customized_bind_dict(arg_bind_core, available_cpus, "CORE")
        device_to_cpu_map = arg_bind_core
    elif arg_bind_core is True:
        device_to_cpu_map = _generate_auto_bind_core_strategy(local_worker_num, available_cpus)

    if arg_bind_numa:
        if not _check_command_availability("numactl"):
            logger.warning("Command 'numactl' is not supported, arg '--bind_numa' takes no effect!")
            arg_bind_numa = False
        elif not available_numas:
            logger.warning("Failed to acquire available numa node info from '/sys/devices/system/node/'. "
                           "Will not launch process with numactl.")
            arg_bind_numa = False

    if isinstance(arg_bind_numa, dict):
        _validate_customized_bind_dict(arg_bind_numa, available_numas, "NUMA")
        device_to_numa_map = arg_bind_numa
    elif arg_bind_numa is True:
        device_to_numa_map = _generate_auto_bind_numa_strategy(local_worker_num, available_numas)

    bind_context = BindContext()
    bind_context.bind_numa = arg_bind_numa
    bind_context.bind_core = arg_bind_core
    bind_context.available_numas = available_numas
    bind_context.available_cpus = available_cpus
    bind_context.device_to_numa_map = device_to_numa_map
    bind_context.device_to_cpu_map = device_to_cpu_map

    return bind_context


def _parse_global_device_to_cpu_map(local_rank_id, physical_device_id, device_to_cpu_map):
    """
    Parse the global device_to_cpu_map and return a cpu list for assigned local_rank_id.
    """
    filtered_map = {k: v for k, v in device_to_cpu_map.items() if k != "scheduler"}
    devices = list(filtered_map.keys())
    cpu_ranges = list(filtered_map.values())

    if local_rank_id >= len(devices):
        logger.warning(f"Cannot find process[{local_rank_id}] in args '--bind_core'. "
                       "Will not launch process with taskset.")
        return ""
    input_device_id = int(devices[local_rank_id].replace("device", ""))
    if physical_device_id != input_device_id:
        logger.warning(f"Cannot find physical_device_id[{physical_device_id}] for process[{local_rank_id}] "
                       "in args '--bind_core'. Will not launch process with taskset.")
        return ""
    worker_cpu_list = cpu_ranges[local_rank_id]
    worker_cpu_str = ",".join(map(str, worker_cpu_list))
    return worker_cpu_str


def _parse_global_device_to_numa_map(local_rank_id, physical_device_id, device_to_numa_map):
    """
    Parse the global device_to_numa_map and return a NUMA node list for assigned local_rank_id.
    """
    filtered_map = {k: v for k, v in device_to_numa_map.items() if k != "scheduler"}
    devices = list(filtered_map.keys())
    numa_ranges = list(filtered_map.values())

    if local_rank_id >= len(devices):
        logger.warning(f"Cannot find process[{local_rank_id}] in args '--bind_numa'. "
                       "Will not launch process with numactl.")
        return ""
    input_device_id = int(devices[local_rank_id].replace("device", ""))
    if physical_device_id != input_device_id:
        logger.warning(f"Cannot find physical_device_id[{physical_device_id}] for process[{local_rank_id}] "
                       "in args '--bind_numa'. Will not launch process with numactl.")
        return ""
    worker_numa_list = numa_ranges[local_rank_id]
    worker_numa_str = ",".join(map(str, worker_numa_list))
    return worker_numa_str


def _generate_auto_bind_core_strategy(local_worker_num, available_cpus):
    """
    Get device to core range assigned for the all processes.
    """
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()

    if not simulation_level:
        device_to_cpu_map = _auto_generate_bind_core_strategy(local_worker_num, available_cpus)
    else:
        device_to_cpu_map = _equal_distribute_cpu_strategy(local_worker_num, available_cpus)

    return device_to_cpu_map


def _generate_auto_bind_numa_strategy(local_worker_num, available_numas):
    """
    Get device to NUMA node assigned for the all processes.
    """
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()

    if len(available_numas) < local_worker_num:
        logger.warning(
            f"Available NUMA nodes count({available_numas}) is insufficient for device count({local_worker_num}. "
            "To avoid performance degradation, will not launch process with numactl automatically. "
            "To enable numactl, please provide a custom binding configuration manually."
        )
        return {}

    if not simulation_level:
        device_to_numa_map = _auto_generate_bind_numa_strategy(local_worker_num, available_numas)
    else:
        device_to_numa_map = _equal_distribute_numa_strategy(local_worker_num, available_numas)

    return device_to_numa_map


def ranges_to_str(num_list):
    """
    Convert a num list to a range string.
    """
    ranges = []
    start = num_list[0]
    for i in range(1, len(num_list)):
        if num_list[i] != num_list[i-1] + 1:
            ranges.append((start, num_list[i-1]))
            start = num_list[i]
    ranges.append((start, num_list[-1]))

    parts = []
    for start, end in ranges:
        if start == end:
            parts.append(str(start))
        else:
            parts.append(f"{start}-{end}")
    return ",".join(parts)


def _generate_bind_core_strategy(local_rank_id, device_to_cpu_map, arg_bind_core):
    """
    Generate CPU core binding strategy: map local rank ID to corresponding CPU.
    """
    physical_device_id = -1
    scheduler_cpu_str = ""
    worker_cpu_str = ""
    cpu_list_for_device = []
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()

    # Scheduler process's local_rank_id is set to -1.
    if local_rank_id == -1:
        scheduler_cpu_list = arg_bind_core.get("scheduler", [])
        scheduler_cpu_str = ",".join(map(str, scheduler_cpu_list))
        return scheduler_cpu_str if scheduler_cpu_str else None

    try:
        physical_device_id = _get_physical_device_id(local_rank_id, simulation_level)
    except RuntimeError as e:
        logger.warning(f"Failed to acquire device id, error: {e} Will not launch process with taskset.")
        return None

    if isinstance(arg_bind_core, dict):
        worker_cpu_str = _parse_global_device_to_cpu_map(local_rank_id, physical_device_id, arg_bind_core)
        if not worker_cpu_str:
            return None
    elif arg_bind_core is True:
        cpu_list_for_device = device_to_cpu_map.get(physical_device_id, [])
        if not cpu_list_for_device:
            return None
        os.environ["MSRUN_CPU_LIST"] = str(cpu_list_for_device)
        worker_cpu_str = ranges_to_str(cpu_list_for_device)
    return worker_cpu_str


def _generate_bind_numa_strategy(local_rank_id, device_to_numa_map, arg_bind_numa, available_numas, available_cpus):
    """
    Generate NUMA node binding strategy: map local rank ID to corresponding NUMA node.
    """
    physical_device_id = -1
    scheduler_numa_str = ""
    worker_numa_str = ""
    numa_list_for_device = []
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()

    # Scheduler process's local_rank_id is set to -1.
    if local_rank_id == -1:
        scheduler_numa_list = arg_bind_numa.get("scheduler", [])
        scheduler_numa_str = ",".join(map(str, scheduler_numa_list))
        return scheduler_numa_str if scheduler_numa_str else None

    try:
        physical_device_id = _get_physical_device_id(local_rank_id, simulation_level)
    except RuntimeError as e:
        logger.warning(f"Failed to acquire device id, error: {e} Will not launch process with numactl.")
        return None

    if isinstance(arg_bind_numa, dict):
        worker_numa_str = _parse_global_device_to_numa_map(local_rank_id, physical_device_id, arg_bind_numa)
        if not worker_numa_str:
            return None
    elif arg_bind_numa is True:
        numa_list_for_device = device_to_numa_map.get(physical_device_id, [])
        if not numa_list_for_device:
            return None
        try:
            numa_to_cpu_map = _get_cpu_info(available_numas, available_cpus)
            cpu_list_for_device = numa_to_cpu_map[int(numa_list_for_device[0])]
            if cpu_list_for_device:
                os.environ["MSRUN_CPU_LIST"] = str(cpu_list_for_device)
        except RuntimeError as e:
            logger.warning(f"MSRUN_CPU_LIST is not set by msrun --bind_numa. Error: {e}")
        worker_numa_str = ranges_to_str(numa_list_for_device)
    return worker_numa_str
