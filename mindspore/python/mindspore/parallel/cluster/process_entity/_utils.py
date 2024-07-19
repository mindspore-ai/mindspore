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
import mindspore.log as logger

def _generate_cmd(cmd, cmd_args, output_name):
    """
    Generates a command string to execute a Python script in the background, r
    edirecting the output to a log file.

    """
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaulty use 'python' to launch the job.
        command = f"python {cmd} {' '.join(cmd_args)} > {output_name} 2>&1 &"
    else:
        command = f"{cmd} {' '.join(cmd_args)} > {output_name} 2>&1 &"
    return command


def _generate_cmd_args_list(cmd, cmd_args):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaulty use 'python' to launch the job.
        return ['python'] + [cmd] + cmd_args
    return [cmd] + cmd_args


def _generate_cmd_args_list_with_core(cmd, cmd_args, cpu_start, cpu_end):
    """
    Generates arguments list for 'Popen'. It consists of a binary file name and subsequential arguments.
    """
    # Bind cpu cores to this process.
    taskset_args = ['taskset'] + ['-c'] + [str(cpu_start) + '-' + str(cpu_end)]
    final_cmd = []
    if cmd not in ['python', 'pytest', 'python3']:
        # If user don't set binary file name, defaulty use 'python' to launch the job.
        final_cmd = taskset_args + ['python'] + [cmd] + cmd_args
    else:
        final_cmd = taskset_args + [cmd] + cmd_args
    logger.info(f"Launch process with command: {' '.join(final_cmd)}")
    return final_cmd


def _generate_url(addr, port):
    """
    Generates a url string by addr and port

    """
    url = f"http://{addr}:{port}/"
    return url


def _is_local_ip(ip_address):
    """
    Check if the current input IP address is a local IP address.

    """
    p = os.popen("ip -j addr")
    addr_info_str = p.read()
    p.close()
    if not addr_info_str:
        # This means this host has no "ip -j addr" command.
        # We use socket module to get local ip address.
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((ip_address, 0))
        current_ip = s.getsockname()[0]
        s.close()
        return current_ip == ip_address

    addr_infos = json.loads(addr_info_str)
    for info in addr_infos:
        for addr in info["addr_info"]:
            if addr["local"] == ip_address:
                logger.info(f"IP address found on this node. Address info:{addr}. Found address:{ip_address}")
                return True
    return False


def _send_scale_num(url, scale_num):
    """
    Send an HTTP request to a specified URL, informing scale_num.

    """
    return ""
