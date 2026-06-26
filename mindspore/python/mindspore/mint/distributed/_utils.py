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
"""Utils for mint.distributed"""
import re
import mindspore.log as logger


MIN_VALID_PORT = 1
MAX_VALID_PORT = 65535


def _is_valid_ipv4(ip):
    """
    Validates whether a string is a valid IPv4 address.
    """
    pattern = r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return re.match(pattern, ip) is not None


def _parse_tcp_url_for_ipv4(url):
    """
    Parses a TCP URL (e.g., "tcp://100.100.1.1:1234") to extract the IPv4 address and port number.

    Args:
        url: The TCP URL string to be parsed.

    Returns:
        A tuple containing (ip_address, port).

    Raises:
        ValueError: If the URL format is incorrect, missing necessary parts, contains a non-numeric port,
                    or the port number is out of range.
        TypeError: If the provided `url` is not a string type.
    """
    if not isinstance(url, str):
        raise TypeError("URL must be a string.")

    try:
        prefix = "tcp://"

        if not url.startswith(prefix):
            raise ValueError("must start with 'tcp://'")

        host_port = url[len(prefix):]
        if not host_port:
            raise ValueError("missing IP and port")

        colon_pos = host_port.find(':')
        if colon_pos == -1:
            raise ValueError("missing port separator ':'")

        if ':' in host_port[colon_pos + 1:]:
            raise ValueError("IPv6 addresses are not supported")

        ip = host_port[:colon_pos]
        if not ip:
            raise ValueError("missing IP address")
        if not _is_valid_ipv4(ip):
            raise ValueError(f"invalid IPv4 address - {ip}")

        port_str = host_port[colon_pos + 1:]
        if not port_str:
            raise ValueError("missing port number")
        if not port_str.isdigit():
            raise ValueError(f"port must be numeric - {port_str}")

        port = int(port_str)
        if not MIN_VALID_PORT <= port <= MAX_VALID_PORT:
            raise ValueError(f"port out of range [{MIN_VALID_PORT}, {MAX_VALID_PORT}] - {port}")

        return ip, port

    except ValueError as e:
        logger.error(f"Invalid TCP URL: {e}")
        raise ValueError(f"Failed to parse TCP URL '{url}': {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error parsing TCP URL: {e}")
        raise
