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
import requests
import netifaces

def _generate_cmd(cmd, cmd_args, output_name):
    """
    Generates a command string to execute a Python script in the background, r
    edirecting the output to a log file.

    """
    command = f"python {cmd} {' '.join(cmd_args)} > {output_name}.log 2>&1 &"
    return command

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
    interfaces = netifaces.interfaces()
    for interface in interfaces:
        addresses = netifaces.ifaddresses(interface)
        if netifaces.AF_INET in addresses:
            for addr_info in addresses[netifaces.AF_INET]:
                if addr_info['addr'] == ip_address:
                    return True
        if netifaces.AF_INET6 in addresses:
            for addr_info in addresses[netifaces.AF_INET6]:
                if addr_info['addr'] == ip_address:
                    return True
    return False

def _send_scale_num(url, scale_num):
    """
    Send an HTTP request to a specified URL, informing scale_num.

    """
    try:
        response = requests.post(url, data={"scale_num": scale_num}, timeout=100)
        response.raise_for_status()
        response_data = response.json()
        response_bool = bool(response_data)
        return response_bool
    except requests.exceptions.RequestException:
        return None

def _get_status_and_params(url):
    """
    Send an HTTP request to a specified URL to query status and retrieve partial parameters.

    """
    try:
        response = requests.get(url, timeout=100)
        response.raise_for_status()
        response_data = response.json()
        network_changed = response_data.get("network_changed")
        total_nodes = response_data.get("total_nodes")
        local_nodes = response_data.get("local_nodes")
        return network_changed, total_nodes, local_nodes
    except requests.exceptions.RequestException:
        return None
