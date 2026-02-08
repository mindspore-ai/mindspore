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
import mindspore.log as logger
from mindspore.runtime.thread_bind_core import _get_physical_device_id, _get_cpu_available, \
    _auto_generate_strategy, _equal_distribution_strategy

CURRENT_IP = None


def _generate_cmd(cmd, cmd_args, local_rank, device_to_cpu_map, arg_bind_core):
    """
    Generates a command string to execute a Python script in the background.

    """
    if local_rank == -1 and not isinstance(arg_bind_core, dict):
        arg_bind_core = False

    if not arg_bind_core:
        return _generate_cmd_args_list(cmd, cmd_args)

    affinity_cpu_str = _generate_bind_core_strategy(local_rank, device_to_cpu_map, arg_bind_core)
    if affinity_cpu_str is not None:
        return _generate_cmd_args_list_with_core(cmd, cmd_args, affinity_cpu_str)

    return _generate_cmd_args_list(cmd, cmd_args)


def _generate_cmd_args_list(cmd, cmd_args):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaultly use 'python' to launch the job.
        return ['python'] + [cmd] + cmd_args
    return [cmd] + cmd_args


def _generate_cmd_args_list_with_core(cmd, cmd_args, affinity_cpu_str):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    # Bind cpu cores to this process.
    taskset_args = ['taskset'] + ['-c'] + [affinity_cpu_str]
    final_cmd = []
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaultly use 'python' to launch the job.
        final_cmd = taskset_args + ['python'] + [cmd] + cmd_args
    else:
        final_cmd = taskset_args + [cmd] + cmd_args
    return final_cmd


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


def _generate_auto_bind_core_strategy(local_worker_num):
    """
    Get device to core range assigned for the all processes.

    """
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()

    try:
        available_cpus = _get_cpu_available()
    except RuntimeError as e:
        logger.warning(f"Failed to acquire available cpu info, error: {e} Will not launch process with taskset.")
        return {}

    if not simulation_level:
        device_to_cpu_map = _auto_generate_strategy(local_worker_num, available_cpus)
    else:
        device_to_cpu_map = _equal_distribution_strategy(local_worker_num, available_cpus)

    return device_to_cpu_map


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
    Get device to core range assigned for the all processes.

    """
    physical_device_id = -1
    affinity_cpu_str = ""
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
        affinity_cpu_str = _parse_global_device_to_cpu_map(local_rank_id, physical_device_id, arg_bind_core)
        if not affinity_cpu_str:
            return None
    elif arg_bind_core is True:
        cpu_list_for_device = device_to_cpu_map.get(physical_device_id, [])
        if not cpu_list_for_device:
            return None
        os.environ["MSRUN_CPU_LIST"] = str(cpu_list_for_device)
        affinity_cpu_str = ranges_to_str(cpu_list_for_device)
    return affinity_cpu_str
