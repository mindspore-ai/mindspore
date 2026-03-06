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
from ._bind import (
    _generate_bind_core_strategy,
    _generate_bind_numa_strategy,
    _generate_bind_prefix_from_config,
    _is_new_bind_config,
)

CURRENT_IP = None


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

    # If a binding config file is passed in, generate the corresponding command based on the JSON file.
    if _is_new_bind_config(arg_bind_numa):
        binding_prefix = _generate_bind_prefix_from_config(local_rank, arg_bind_numa)
        return binding_prefix + base_cmd

    # --- NUMA Binding Logic ---
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
            s.connect((ip_address, 8117))
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
