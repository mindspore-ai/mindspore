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

CURRENT_IP = {}


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


def _normalize_ip(ip_address):
    """
    Normalize ip format:
    1. IPv4 keeps unchanged.
    2. IPv6 allows input with square brackets and removes them for system calls.
    """
    if not isinstance(ip_address, str):
        return ip_address
    ip_addr = ip_address.strip()
    if ip_addr.startswith('[') and ip_addr.endswith(']'):
        return ip_addr[1:-1]
    return ip_addr


def _get_ip_family(ip_address):
    """
    Parse ip address and return `(ip_obj, family)`.
    """
    normalized_ip = _normalize_ip(ip_address)
    ip_obj = ipaddress.ip_address(normalized_ip)
    family = socket.AF_INET6 if ip_obj.version == 6 else socket.AF_INET
    return ip_obj, family


def _is_same_ip(lhs_ip, rhs_ip):
    """
    Compare two ip strings by semantic equality instead of raw string form.
    """
    try:
        return ipaddress.ip_address(_normalize_ip(lhs_ip)) == ipaddress.ip_address(_normalize_ip(rhs_ip))
    except ValueError:
        return False


def _get_addr_infos():
    """
    Get network interface info from `ip -j addr`.
    """
    try:
        p = os.popen("ip -j addr")
        addr_info_str = p.read()
        p.close()
    except OSError as e:
        logger.warning(f"Failed to execute `ip -j addr`: {e}")
        return []
    if not addr_info_str:
        return []
    try:
        return json.loads(addr_info_str)
    except json.JSONDecodeError:
        # The command may be unavailable or unsupported and print plain text error.
        logger.warning("The output of `ip -j addr` is not valid json, fallback to empty interface info.")
        return []


def _get_interface_name_by_ip(ip_address):
    """
    Find interface name by local interface ip.
    """
    normalized_ip = _normalize_ip(ip_address)
    for info in _get_addr_infos():
        for addr in info.get("addr_info", []):
            if _is_same_ip(addr.get("local", ""), normalized_ip):
                return info.get("ifname", "")
    return ""


def _get_scope_id_from_ip(ip_address):
    """
    Get IPv6 scope id based on worker ip.
    """
    normalized_ip = _normalize_ip(ip_address)
    if not normalized_ip:
        return 0
    if_name = _get_interface_name_by_ip(normalized_ip)
    if not if_name:
        return 0
    try:
        return socket.if_nametoindex(if_name)
    except OSError:
        return 0


def _get_local_ip(ip_address):
    """
    Get current local source ip used to reach `ip_address`.

    Notes:
    - Input IPv6 may be wrapped by `[]`, which is normalized before socket calls.
    - For IPv6 link-local targets, a valid scope id (interface index) may be required.
    - Results are cached by `(address family, destination ip)` to avoid repeated probes.
    """
    normalized_ip = _normalize_ip(ip_address)
    ip_obj, family = _get_ip_family(normalized_ip)
    cache_key = f"{family}:{normalized_ip}"
    if cache_key not in CURRENT_IP:
        try:
            if family == socket.AF_INET6:
                scope_ids = [0]
                # Link-local IPv6 usually requires scope id (interface index).
                # We first infer it from interface name -> index.
                if ip_obj.is_link_local:
                    scope_id = _get_scope_id_from_ip(normalized_ip)
                    if scope_id > 0:
                        scope_ids = [scope_id]
                    else:
                        # Fallback: try all local interface indexes when we cannot infer one.
                        scope_ids = [idx for idx, _ in socket.if_nameindex()]

                # Probe which scope id works by connecting UDP socket.
                # No packets are sent for this connect; we only learn local bound address.
                last_error = None
                for scope_id in scope_ids:
                    s = socket.socket(family, socket.SOCK_DGRAM)
                    try:
                        s.connect((normalized_ip, 8117, 0, scope_id))
                        CURRENT_IP[cache_key] = s.getsockname()[0]
                        last_error = None
                        break
                    except OSError as e:
                        last_error = e
                    finally:
                        s.close()
                if last_error is not None:
                    raise last_error
            else:
                s = socket.socket(family, socket.SOCK_DGRAM)
                s.connect((normalized_ip, 8117))
                CURRENT_IP[cache_key] = s.getsockname()[0]
                s.close()
        except Exception as e:
            raise RuntimeError("Get local ip has failed. Please verify that the accessible address has been "
                               "specified in the '--master_address' parameter") from e
    return CURRENT_IP[cache_key]


def _is_local_ip(ip_address):
    """
    Check if the current input IP address is a local IP address.
    """
    normalized_ip = _normalize_ip(ip_address)
    addr_infos = _get_addr_infos()

    for info in addr_infos:
        for addr in info.get("addr_info", []):
            if _is_same_ip(addr.get("local", ""), normalized_ip):
                logger.info(f"IP address found on this node. Address info:{addr}. Found address:{ip_address}")
                return True

    # Fallback path: if local address list is unavailable or no direct hit, infer by routing result.
    current_ip = _get_local_ip(ip_address)
    return _is_same_ip(current_ip, normalized_ip)


def _convert_addr_to_ip(master_addr):
    """
    Check whether the input parameter 'master_addr' is an IP address.
    If a hostname is inserted, it will be converted to IP and then set as master host's IP.
    IPv6 input supports square bracket format like "[2001:db8::1]".
    """
    normalized_addr = _normalize_ip(master_addr)
    try:
        ip_obj = ipaddress.ip_address(normalized_addr)
        return str(ip_obj)
    except (ipaddress.AddressValueError, ValueError):
        pass

    try:
        ip_address = socket.getaddrinfo(normalized_addr, None, socket.AF_UNSPEC, socket.SOCK_STREAM)[0][4][0]
        logger.info(f"Convert input host name:{master_addr} to ip address:{ip_address}.")
        return ip_address
    except (socket.gaierror, ValueError, TypeError, IndexError, OSError) as e:
        raise RuntimeError("DNS resolution has failed. Please verify that the correct hostname has been "
                           "specified in the '--master_address' parameter") from e


def _send_scale_num(url, scale_num):
    """
    Send an HTTP request to a specified URL, informing scale_num.
    """
    return ""
