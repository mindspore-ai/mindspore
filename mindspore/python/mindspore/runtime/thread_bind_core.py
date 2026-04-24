# Copyright 2024 Huawei Technologies Co., Ltd
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

"""Executor manager interfaces."""
import json
import subprocess
import re
import os
import ast
from itertools import chain
from mindspore import log as logger
from mindspore import context
from mindspore.communication import get_local_rank_size


def execute_command(cmd_list, timeout=10):
    """
    Execute a system command and return its output.

    Args:
        cmd_list (list): A list of strings representing the command and its arguments.
        timeout (second): Timeout for executing command.

    Returns:
        str: The decoded standard output from the command execution.
    """
    cmd_str = " ".join(cmd_list)
    try:
        with subprocess.Popen(
            cmd_list,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict"
        ) as p:
            try:
                out, err = p.communicate(timeout=timeout)
            except subprocess.TimeoutExpired as e:
                p.kill()
                raise RuntimeError(f"Command '{cmd_str}' timed out after {timeout}s!") from e

            if p.returncode != 0:
                raise RuntimeError(f"Command '{cmd_str}' failed (return code {p.returncode})! Stderr: {err.strip()}")
        return out
    except FileNotFoundError as e:
        raise RuntimeError(f"Command '{cmd_str}' not found!") from e
    except PermissionError as e:
        raise RuntimeError(f"Permission denied to execute '{cmd_str}'!") from e
    except OSError as e:
        raise RuntimeError(f"Command '{cmd_str}' failed to start (system error): {str(e)} Possible causes: missing "
                           "dependent libraries, insufficient system resources, etc.") from e
    except Exception as e:
        raise RuntimeError(f"Failed to execute '{cmd_str}'! {e}") from e


def _adapt_to_dict(affinity_cpu_list):
    """
    Adapt to dict type affinity_cpu_list.
    """
    if not isinstance(affinity_cpu_list, dict):
        return affinity_cpu_list

    logical_device_id = context.get_context("device_id")
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()
    physical_device_id = _get_physical_device_id(logical_device_id, simulation_level)
    device_key = f"device{physical_device_id}"
    return affinity_cpu_list.get(device_key, False)


def _validate_affinity_cpu_list(affinity_cpu_list):
    """
    Validate the user-configured affinity_cpu_list.

    Args:
        affinity_cpu_list (list): Customized bind-core strategy to be validated.

    Returns:
        None.
    """
    if affinity_cpu_list is None:
        return

    if not isinstance(affinity_cpu_list, list):
        raise TypeError(f"The parameter '{affinity_cpu_list}' must be list, but got {type(affinity_cpu_list)}")

    range_pattern = re.compile(r'^(\d+-\d+|\d+)(,(\d+-\d+|\d+))*$')

    for cpu_range in affinity_cpu_list:
        if not isinstance(cpu_range, str):
            raise ValueError(f"CPU range '{cpu_range}' in '{affinity_cpu_list}' should be a string.")
        if not range_pattern.match(cpu_range):
            raise ValueError(f"CPU range '{cpu_range}' in '{affinity_cpu_list}' should be in format "
                             "'cpuidX-cpuidY' or 'cpuidX,cpuidY,cpuidZ'.")


def _validate_module_cpu_index(module_to_cpu_dict):
    """
    Validate the user-configured module_to_cpu_dict.

    Args:
        module_to_cpu_dict (dict): Customized module-to-CPU mapping to be validated.

    Returns:
        None.
    """
    if module_to_cpu_dict is None:
        return

    if not isinstance(module_to_cpu_dict, dict):
        raise TypeError(f"The parameter '{module_to_cpu_dict}' must be dict, but got {type(module_to_cpu_dict)}")

    for module_name, cpu_indices in module_to_cpu_dict.items():
        if not isinstance(cpu_indices, list):
            raise ValueError(f"The value of module_to_cpu_dict: {cpu_indices} should be a list.")
        for cpu_index in cpu_indices:
            if not isinstance(cpu_index, int) or cpu_index < 0:
                raise ValueError(f"CPU index '{cpu_index}' for module '{module_name}' in '{cpu_indices}' "
                                 "should be a non-negative integer.")


def _get_available_cpus():
    """
    Get available CPU IDs from cgroup cpuset file.
    
    Returns:
        list[int]: Sorted list of available CPU IDs (empty list if failed)
    """
    cpu_set_path = "/sys/fs/cgroup/cpuset/cpuset.cpus"
    cpu_pattern = re.compile(r'(\d+)(?:-(\d+))?')
    available_cpus = []

    try:
        with open(cpu_set_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except OSError as e:
        logger.warning(f"Failed to read cpuset file {cpu_set_path}: {e}")
        return []

    if not content:
        return []

    try:
        for segment in content.split(","):
            segment = segment.strip()
            if not segment:
                continue

            match = cpu_pattern.fullmatch(segment)
            if not match:
                return []

            start, end = match.groups()
            start = int(start)
            end = int(end) if end else start

            if start > end:
                return []
            available_cpus.extend(range(start, end + 1))
    except (ValueError, TypeError):
        return []

    return sorted(list(set(available_cpus)))


def _expand_ranges(ranges):
    """
    Expand ranges/ids to a sorted unique id list.

    Args:
        ranges (str or int): e.g. "0-4,6,9-10" or 3

    Returns:
        list[int]: Sorted, unique ids.
    """
    if ranges is None:
        return []

    cpu_ids = []
    if isinstance(ranges, int):
        cpu_ids.append(ranges)
    elif isinstance(ranges, str):
        if ",," in ranges or ranges.startswith(",") or ranges.endswith(","):
            raise ValueError(f"CPU range '{ranges}' is invalid due to empty segment.")
        segments = [seg.strip() for seg in ranges.split(",") if seg.strip()]
        for seg in segments:
            if "-" in seg:
                endpoints = seg.split("-")
                if len(endpoints) != 2:
                    raise ValueError(f"CPU range '{seg}' is invalid.")
                start, end = int(endpoints[0]), int(endpoints[1])
                if start > end:
                    raise ValueError(f"CPU range '{seg}' start should not exceed end.")
                cpu_ids.extend(range(start, end + 1))
            else:
                cpu_ids.append(int(seg))
    else:
        raise TypeError(f"The parameter '{ranges}' must be str/int, but got {type(ranges)}")
    return sorted(set(cpu_ids))


def _int_list_to_range_str(num_list):
    """
    Convert a CPU id list to condensed range string.
    """
    if not num_list:
        return ""
    ranges = []
    start = num_list[0]
    for i in range(1, len(num_list)):
        if num_list[i] != num_list[i - 1] + 1:
            ranges.append((start, num_list[i - 1]))
            start = num_list[i]
    ranges.append((start, num_list[-1]))

    parts = []
    for s_val, e_val in ranges:
        if s_val == e_val:
            parts.append(str(s_val))
        else:
            parts.append(f"{s_val}-{e_val}")
    return ",".join(parts)


def _parse_device_key(key):
    """
    Parse device key and extract device ID.
    Supports both 'deviceX' and 'device_X' formats.

    Args:
        key (str): Device key string.

    Returns:
        int or None: Device ID if valid device key, None otherwise.
    """
    if not isinstance(key, str):
        return None
    # Match 'deviceX' or 'device_X' format
    match = re.match(r'^device_?(\d+)$', key)
    if match:
        return int(match.group(1))
    return None


def _normalize_and_validate_device_keys(bind_dict):
    """
    Normalize device keys to 'deviceX' format and validate them.
    Converts 'device_X' format to 'deviceX' format in-place, then validates:
    1. All keys are valid device keys or 'scheduler'
    2. Device IDs don't exceed available device count

    Args:
        bind_dict (dict): Dictionary with device keys.

    Raises:
        ValueError: If invalid keys are found or device IDs exceed available count.
    """
    if not isinstance(bind_dict, dict):
        return

    # Step 1: Normalize keys (convert 'device_X' to 'deviceX')
    keys_to_rename = {}
    for key in bind_dict.keys():
        if key == "scheduler":
            continue
        device_id = _parse_device_key(key)
        if device_id is None:
            raise ValueError(
                f"Invalid key '{key}'. "
                "Only 'deviceX', 'device_X' (where X is a number), and 'scheduler' are supported."
            )
        if "_" in key:
            # Convert 'device_X' to 'deviceX'
            keys_to_rename[key] = f"device{device_id}"

    # Rename keys
    for old_key, new_key in keys_to_rename.items():
        bind_dict[new_key] = bind_dict.pop(old_key)

    # Step 2: Validate device IDs don't exceed available count
    try:
        _, available_devices = _get_device_map_info()
        if not available_devices:
            # If no devices detected, skip validation
            return
        max_device_id = max(available_devices)
    except RuntimeError:
        # If we can't get device info, skip device count validation
        return

    for key in bind_dict.keys():
        if key == "scheduler":
            continue
        device_id = _parse_device_key(key)
        if device_id is not None and max_device_id is not None and device_id > max_device_id:
            raise ValueError(
                f"Device ID {device_id} in key '{key}' exceeds available device count. "
                f"Maximum available device ID is {max_device_id}."
            )


def _load_bind_file(bind_file):
    """
    Load and validate a bind json file.
    """
    if not isinstance(bind_file, str):
        raise TypeError(f"The parameter 'bind_file' must be a string path, but got {type(bind_file)}")
    if not os.path.isfile(bind_file):
        raise ValueError(f"The bind file '{bind_file}' does not exist.")
    with open(bind_file, "r", encoding="utf-8") as fp:
        try:
            content = json.load(fp)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse bind file '{bind_file}' as JSON.") from e
    if not isinstance(content, dict):
        raise ValueError(f"The bind file '{bind_file}' content must be a JSON object.")
    return content


def _validate_bind_config(bind_config, bind_file):
    """
    Validate bind_file's input bind_cpu_mode/bind_memory_mode/actor_thread_fix_bind.
    """
    if not isinstance(bind_config, dict):
        raise ValueError(f"The 'bind_config' field in bind file '{bind_file}' must be an object.")
    cpu_mode = str(bind_config.get("bind_cpu_mode", "none")).lower()
    if cpu_mode not in {"cpu", "numa", "none"}:
        raise ValueError("'bind_cpu_mode' only supports 'cpu'/'numa'/'none'.")
    mem_mode = str(bind_config.get("bind_memory_mode", "none")).lower()
    if mem_mode not in {"numa", "none"}:
        raise ValueError("'bind_memory_mode' only supports 'numa'/'none'.")
    if "actor_thread_fix_bind" in bind_config:
        atfb = bind_config.get("actor_thread_fix_bind")
        if not isinstance(atfb, bool):
            raise ValueError("'actor_thread_fix_bind' only supports bool type True/False.")
    return bind_config


def _extract_affinity_from_bind_file(bind_file):
    """
    Extract affinity_cpu_list and module_to_cpu_dict from a bind json file for current device.
    """
    content = _load_bind_file(bind_file)
    bind_config = _validate_bind_config(content.get("bind_config", {}), bind_file)
    fix_bind = bind_config.get("actor_thread_fix_bind", None)
    if fix_bind is None:
        actor_thread_fix_bind = ""
    else:
        actor_thread_fix_bind = "True" if fix_bind else "False"
    cpu_mode = str(bind_config.get("bind_cpu_mode", "none")).lower()
    if cpu_mode != "cpu":
        logger.warning(f"bind_file '{bind_file}' is provided but 'bind_cpu_mode' is not 'cpu'. "
                       "Skip enabling set_cpu_affinity based on bind_file.")
        return None, None, None
    bind_cpu = content.get("bind_cpu", {})
    if not isinstance(bind_cpu, dict):
        raise ValueError(f"The 'bind_cpu' field in bind file '{bind_file}' must be a dict.")

    # Normalize and validate device keys
    _normalize_and_validate_device_keys(bind_cpu)

    logical_device_id = context.get_context("device_id")
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()
    physical_device_id = _get_physical_device_id(logical_device_id, simulation_level)
    device_key = f"device{physical_device_id}"
    device_cpu_cfg = bind_cpu.get(device_key)
    if device_cpu_cfg is None:
        logger.warning(f"bind_file '{bind_file}' has no 'bind_cpu' config for {device_key}, "
                       "Skip enabling set_cpu_affinity.")
        return None, None, None
    if not isinstance(device_cpu_cfg, dict):
        logger.warning(f"bind_file '{bind_file}' entry for {device_key}, type of {device_cpu_cfg} should be dict, "
                       f"but got {type(device_cpu_cfg)}. Skip enabling set_cpu_affinity.")
        return None, None, None

    module_cpu_map = {}
    for module, ranges in device_cpu_cfg.items():
        if not isinstance(ranges, (str, int)):
            raise ValueError(f"The 'bind_cpu' entry for {module}, type of {ranges} should be string/int, "
                             f"but got {type(ranges)}.")
        module_cpu_map[module] = _expand_ranges(ranges)

    all_cpus = sorted(set(chain.from_iterable(module_cpu_map.values())))
    if not all_cpus:
        logger.warning(f"bind_file '{bind_file}' has empty cpu list for {device_key}. "
                       "Skip enabling set_cpu_affinity.")
        return None, None, None

    affinity_cpu_list = [_int_list_to_range_str(all_cpus)]
    cpu_to_index = {cpu_id: idx for idx, cpu_id in enumerate(all_cpus)}
    module_to_cpu_dict = {
        module: [cpu_to_index[cpu] for cpu in cpu_ids if cpu in cpu_to_index]
        for module, cpu_ids in module_cpu_map.items() if module != "main"
    }
    # When msrun doesn't provide a bind file, allow set_cpu_affinity to bind "main".
    if not os.getenv("MSRUN_BIND_FILE", "").strip() and "main" in module_cpu_map:
        module_to_cpu_dict["main"] = [cpu_to_index[cpu] for cpu in module_cpu_map["main"] if cpu in cpu_to_index]
    return affinity_cpu_list, module_to_cpu_dict, actor_thread_fix_bind


def _get_available_numas():
    """
    Get NUMA node IDs that can be used by numactl from sysfs.

    A NUMA node may have CPUs but no local memory. Such memoryless nodes cannot
    be used by numactl safely, so only nodes with both CPU and memory resources
    are returned when sysfs exposes the resource masks.
    
    Returns:
        list[int]: Sorted list of NUMA node IDs (empty list if failed)
    """
    numa_node_path = "/sys/devices/system/node/"

    if not (os.path.isdir(numa_node_path) and os.access(numa_node_path, os.R_OK)):
        return []

    all_numas = _get_sysfs_numa_node_ids(numa_node_path)
    if not all_numas:
        return []

    cpu_numas = _get_sysfs_numa_range_ids(os.path.join(numa_node_path, "has_cpu"))
    memory_numas = _get_sysfs_numa_range_ids(os.path.join(numa_node_path, "has_memory"))

    # Keep old behavior on systems that do not expose resource masks.
    if cpu_numas is None and memory_numas is None:
        return all_numas

    all_numa_set = set(all_numas)
    cpu_numa_set = set(cpu_numas) if cpu_numas is not None else all_numa_set
    memory_numa_set = set(memory_numas) if memory_numas is not None else all_numa_set
    memoryless_numas = sorted((all_numa_set & cpu_numa_set) - memory_numa_set)
    if memoryless_numas:
        logger.warning(f"Detected memoryless NUMA nodes {memoryless_numas}. These nodes have CPUs but no "
                       "local memory, cannot be used by numactl, and will be excluded from NUMA binding.")

    return sorted(all_numa_set & cpu_numa_set & memory_numa_set)


def _get_sysfs_numa_node_ids(numa_node_path):
    """
    Get existing NUMA node IDs from sysfs node directories.
    """
    numa_nodes = []
    try:
        for dir_name in os.listdir(numa_node_path):
            if re.fullmatch(r"node\d+", dir_name):
                node_id = int(dir_name.replace("node", ""))
                numa_nodes.append(node_id)
    except OSError as e:
        logger.warning(f"Failed to read NUMA node directory {numa_node_path}: {e}")

    return sorted(numa_nodes)


def _get_sysfs_numa_range_ids(range_file):
    """
    Read a sysfs node range file such as has_cpu or has_memory.

    Returns:
        list[int] or None: Parsed node IDs, or None when the range file cannot be read or parsed.
    """
    try:
        with open(range_file, "r", encoding="utf-8") as fp:
            content = fp.read().strip()
    except OSError:
        return None

    try:
        return _expand_ranges(content)
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to parse NUMA node range file {range_file}: {e}")
        return None


class DeviceInfo:
    """
    A class to represent information about an Ascend device.

    Attributes:
        npu_id (int): The ID of the NPU.
        chip_id (int): The ID of the chip.
        chip_logic_id (int, str): The logical ID of the chip, which can be an integer or a string.
        chip_name (str): The name of the chip.

    Methods:
        _parse_info_line(info_line): Initializes the attributes by parsing input info_line.
    """
    def __init__(self, info_line):
        self.npu_id = 0
        self.chip_id = 0
        self.chip_logic_id = 0
        self.chip_name = ""
        self._parse_info_line(info_line)

    def _parse_info_line(self, info_line):
        self.npu_id, self.chip_id, self.chip_logic_id, self.chip_name = info_line.strip().split(None, 3)
        self.npu_id = int(self.npu_id)
        self.chip_id = int(self.chip_id)
        if self.chip_logic_id.isnumeric():
            self.chip_logic_id = int(self.chip_logic_id)


def _get_device_map_info():
    """
    Get abbreviated information about all NPUs on the environment.

    Returns:
        dict: Mapping of NPU logical ID to its details.
        set: Contains all available NPU logical ids on the environment.
    """
    device_map_info = {}
    available_devices = set()

    device_map = execute_command(["npu-smi", "info", "-m"]).strip().split("\n")[1:]
    for line in device_map:
        device_info = DeviceInfo(line.strip())
        if isinstance(device_info.chip_logic_id, int):
            device_map_info[device_info.chip_logic_id] = device_info
            available_devices.add(device_info.chip_logic_id)

    return device_map_info, sorted(available_devices)


def _get_pcie_info(device_map_info, available_devices, keyword="PCIeBusInfo"):
    """
    Get the PCIe number of the NPU device.

    Args:
        device_map_info (dict): A map of NPU logical ID to its details.
        available_devices (set): All available NPU logical ids on the environment.

    Returns:
        dict: Mapping of NPU logical ID to its PCIe number.
    """
    device_to_pcie_map = {}

    for device in available_devices:
        device_info = device_map_info.get(device)
        if not device_info:
            raise RuntimeError("Failed to get device pcie info.")
        pcie_info = execute_command(["npu-smi", "info", "-t", "board", "-i", f"{device_info.npu_id}",
                                     "-c", f"{device_info.chip_id}"]).strip().split("\n")
        for _ in pcie_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                device_to_pcie_map[device] = line[len(keyword) + 1:]
                break

    return device_to_pcie_map


def _get_numa_info(device_to_pcie_map, keyword="NUMAnode"):
    """
    Get NUNA node affinity for device based on PCIe.

    Args:
        device_to_pcie_map (dict): A map of NPU logical ID to its PCIe number.

    Returns:
        dict: Mapping of device ID to its affinity NUMA nodes.
        dict: Mapping of NUMA node to its affinity device IDs.
    """
    device_to_numa_map = {}
    numa_to_device_map = {}

    for device, pcie_no in device_to_pcie_map.items():
        numa_info = execute_command(["lspci", "-s", f"{pcie_no}", "-vvv"]).strip().split("\n")
        for _ in numa_info:
            line = ''.join(_.split())
            if line.startswith(keyword):
                numa_id = int(line[len(keyword) + 1:])
                device_to_numa_map[device] = numa_id

                devices = numa_to_device_map.get(numa_id, None)
                if devices is None:
                    numa_to_device_map[numa_id] = []
                numa_to_device_map[numa_id].append(device)
                break

    return device_to_numa_map, numa_to_device_map


def _get_cpu_info(numa_ids, available_cpus, keyword1="NUMAnode", keyword2="CPU(s)"):
    """
    Get information about the CPUs on the NUMA nodes on the environment.

    Args:
        numa_ids (list): A list of NUMA nodes need to get related CPU information.
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of NUMA node to its affinity CPUs.
    """
    numa_to_cpu_map = {}

    cpu_info = execute_command(["lscpu"]).strip().split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if line.startswith(keyword1):
            pattern = re.escape(keyword1) + r'(\d+)' + re.escape(keyword2)
            match = re.search(pattern, line)
            if match:
                numa_id = int(match.group(1))
                split_info = line.split(":")
                cpu_id_ranges = split_info[-1].split(",")
                ranges = []
                for range_str in cpu_id_ranges:
                    endpoints = range_str.split("-")
                    if len(endpoints) != 2:
                        raise RuntimeError("Failed to parse the result of executing 'lscpu'.")
                    ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1) if cid in available_cpus]
                if numa_id not in numa_ids:
                    numa_id = int(-1)
                if numa_id not in numa_to_cpu_map:
                    numa_to_cpu_map[numa_id] = []
                numa_to_cpu_map[numa_id].extend(ranges)

    return numa_to_cpu_map


def _get_physical_device_id(logical_device_id, simulation_level):
    """
    Get physical device id from logical device id.

    Args:
        logical_device_id (int): The logical device id for this process in the task.
        simulation_level (string): Dryrun's simulation level.

    Returns:
        int: The physical device id for this process in the host.
    """
    env_visible_device = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if context.get_context("device_target") == "Ascend" and env_visible_device and not simulation_level:
        list_visible_device = []
        for item in env_visible_device.split(','):
            list_visible_device.append(int(item))
        list_visible_device.sort()
        if logical_device_id >= len(list_visible_device):
            raise RuntimeError("Device id exceeds the number of available devices.")
        physical_device_id = list_visible_device[logical_device_id]
    else:
        physical_device_id = logical_device_id

    return physical_device_id


def _equal_distribute_cpu_strategy(device_count, available_cpus):
    """
    Generate global bind core strategy by equally distributing available cpus.

    Args:
        device_count(int): The total number of device in the task.
        available_cpus(list): A list of cpus in the environment.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    device_to_cpu_map = {}

    total_cpus = len(available_cpus)
    cpu_num_per_device = total_cpus // device_count
    if cpu_num_per_device < 1:
        logger.warning("Available CPUs is less than 1. Will not enable bind core feature.")
        return {}

    for i in range(device_count):
        cpu_start = i * cpu_num_per_device
        cpu_end = (i + 1) * cpu_num_per_device if i != device_count - 1 else total_cpus
        device_to_cpu_map[i] = available_cpus[cpu_start:cpu_end]

    return device_to_cpu_map


def _equal_distribute_numa_strategy(device_count, available_numas):
    """
    Generate global bind numa strategy by equally distributing available NUMA nodes.

    Args:
        device_count(int): The total number of device in the task.
        available_numas(list): A list of NUMA nodes in the environment (sorted).

    Returns:
        dict: Mapping of device index (int) to its affinity NUMA node list (list[int]).
              Empty dict if NUMA nodes are insufficient (less than device count).
    """
    if len(available_numas) < device_count:
        return {}

    device_to_numa_map = {idx: [available_numas[idx]] for idx in range(device_count)}

    return device_to_numa_map


def _assemble_env_info(available_devices, available_cpus, affinity_flag, numa_to_cpu_map, device_to_numa_map):
    """
    Assemble all results of commands based on the hardware on the environment.

    Args:
        available_devices (list): All available NPU logical ids on the environment.
        available_cpus (list): A list of available CPUs on the environment.
        affinity_flag (bool): Whether or not it satisfies generating CPU affinity bind-core
          strategy based on the resources on the environment.
        numa_to_cpu_map (dict): A map of NUMA node to its affinity CPUs.
        device_to_numa_map (dict): A map of device ID to its affinity NUMA nodes.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    device_to_cpu_map = {device_id: [] for device_id in available_devices}
    cpu_num_per_device = len(available_cpus) // len(available_devices)

    if cpu_num_per_device < 1:
        logger.warning("Available CPUs is less than 1. Will not enable bind core feature.")
        return {}

    if affinity_flag:
        device_to_cpu_idx = {numa_id: 0 for numa_id in numa_to_cpu_map}
        for device_id in available_devices:
            # Prioritize the use of affinity cpu resources.
            numa_id = device_to_numa_map.get(device_id)
            affinity_cpu_start_idx = device_to_cpu_idx[numa_id]
            affinity_cpu = numa_to_cpu_map[numa_id][
                affinity_cpu_start_idx: affinity_cpu_start_idx + cpu_num_per_device]
            device_to_cpu_map[device_id].extend(affinity_cpu)
            device_to_cpu_idx[numa_id] = affinity_cpu_start_idx + len(affinity_cpu)

            # If the affinity cpu resources are insufficient then use resources from the non-affinity cpu pool.
            if -1 in device_to_cpu_idx and len(affinity_cpu) < cpu_num_per_device:
                unaffinity_cpu_num = cpu_num_per_device - len(affinity_cpu)
                unaffinity_cpu_start_idx = device_to_cpu_idx[-1]
                unaffinity_cpu = numa_to_cpu_map[-1][
                    unaffinity_cpu_start_idx: unaffinity_cpu_start_idx + unaffinity_cpu_num]
                device_to_cpu_map[device_id].extend(unaffinity_cpu)
                device_to_cpu_idx[-1] = unaffinity_cpu_start_idx + unaffinity_cpu_num
    else:
        # Fall back to average CPU distribution when affinity data is incomplete.
        logger.warning("Affinity binding not applicable, fall back to average CPU distribution.")
        for device_rank, device_id in enumerate(available_devices):
            cpu_start = device_rank * cpu_num_per_device
            device_to_cpu_map[device_id] = available_cpus[cpu_start: cpu_start + cpu_num_per_device]

    return device_to_cpu_map


def _auto_generate_bind_core_strategy(device_count, available_cpus):
    """
    Automatically generate bind-core strategy based on CPU affinity.

    Args:
        device_count(int): The total number of device in the task.
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    device_to_pcie_map = {}
    device_to_numa_map = {}
    numa_to_device_map = {}
    numa_to_cpu_map = {}
    affinity_flag = False

    # Get the hardware resources in the environment. If this fails, will bind core not based on device.
    try:
        device_map_info, available_devices = _get_device_map_info()
    except RuntimeError as e:
        device_to_cpu_map = _equal_distribute_cpu_strategy(device_count, available_cpus)
        logger.warning(f"Failed to acquire device to numa affinity info, from {e} "
                       "Will not bind core based on affinity.")
        return device_to_cpu_map

    # Get the affinity resources in the environment. If this fails, will bind core not based on affinity.
    try:
        device_to_pcie_map = _get_pcie_info(device_map_info, available_devices)
        device_to_numa_map, numa_to_device_map = _get_numa_info(device_to_pcie_map)
        numa_to_cpu_map = _get_cpu_info(list(numa_to_device_map.keys()), available_cpus)
    except RuntimeError as e:
        logger.warning(f"Failed to acquire device to numa affinity info, from {e} "
                       "Will not bind core based on affinity.")
        affinity_flag = False

    if device_to_pcie_map and device_to_numa_map and numa_to_device_map and numa_to_cpu_map:
        affinity_flag = True

    # Auto-generation of bind core strategy for Ascend.
    try:
        device_to_cpu_map = _assemble_env_info(available_devices, available_cpus, affinity_flag,
                                               numa_to_cpu_map, device_to_numa_map)
        return device_to_cpu_map
    except (RuntimeError, ZeroDivisionError) as e:
        logger.warning(f"Failed to auto generate bind core strategy, from {e} "
                       "Will not enable bind core feature.")
        return {}


def _auto_generate_bind_numa_strategy(device_count, available_numas):
    """
    Automatically generate bind-core strategy based on CPU affinity.

    Args:
        device_count(int): The total number of device in the task.
        available_numas(list): A list of NUMA nodes in the environment (sorted).

    Returns:
        dict: Mapping of device to its affinity NUMA nodes.
    """
    numa_count = len(available_numas)
    if numa_count < device_count:
        return {}

    device_to_numa_map = {}

    # Get the hardware resources in the environment. If this fails, will bind numa not based on device.
    try:
        device_map_info, available_devices = _get_device_map_info()
    except RuntimeError as e:
        device_to_numa_map = _equal_distribute_numa_strategy(device_count, available_numas)
        logger.warning(f"Failed to acquire device to numa affinity info, from {e} "
                       "Will not bind numa based on affinity.")
        return device_to_numa_map

    # Get the affinity resources in the environment. If this fails, will bind numa not based on affinity.
    try:
        device_to_pcie_map = _get_pcie_info(device_map_info, available_devices)
        temp_device_to_numa_map, numa_to_device_map = _get_numa_info(device_to_pcie_map)

        # Fall back to average NUMA distribution when affinity data is incomplete.
        if not device_to_pcie_map or not temp_device_to_numa_map or not numa_to_device_map:
            logger.warning("Affinity binding not applicable, fall back to average NUMA distribution.")
            device_to_numa_map = {
                device: [available_numas[i % numa_count]]
                for i, device in enumerate(available_devices)
            }
            return device_to_numa_map

        available_numa_set = set(available_numas)
        invalid_affinity_map = {
            device: numa for device, numa in temp_device_to_numa_map.items()
            if numa not in available_numa_set
        }
        if invalid_affinity_map:
            logger.warning(f"Device affinity NUMA nodes {invalid_affinity_map} are unavailable or memoryless. "
                           f"Only usable NUMA nodes {available_numas} will be used for numactl binding.")

        # Build non-affinity pool (-1) from usable NUMA nodes that have no direct affinity.
        affinity_numas = {
            numa for numa in temp_device_to_numa_map.values()
            if numa in available_numa_set
        }
        numa_to_device_map[-1] = [numa for numa in available_numas if numa not in affinity_numas]

        # If affinity NUMA is unavailable or duplicated, reassign from the non-affinity pool.
        unaffinity_numa_pool = iter(numa_to_device_map.get(-1, []))
        used_numas = set()

        for device, numa in temp_device_to_numa_map.items():
            if numa not in available_numa_set:
                numa = next(unaffinity_numa_pool)
            elif numa in used_numas:
                numa = next(unaffinity_numa_pool)
            used_numas.add(numa)
            device_to_numa_map[device] = [numa]
    except RuntimeError as e:
        logger.warning(f"Failed to acquire device to numa affinity info, from {e} "
                       "Will not bind numa based on affinity.")
        device_to_numa_map = _equal_distribute_numa_strategy(device_count, available_numas)

    return device_to_numa_map


def _customize_generate_strategy(affinity_cpu_list, available_cpus):
    """
    Generate customized bind-core strategy based on user-configured inputs.

    Args:
        affinity_cpu_list (list): User-configured inputs to generate customized bind-core strategy.
        available_cpus (list): A list of available CPUs on the environment.

    Returns:
        dict: Mapping of device to its affinity CPUs.
    """
    cpu_list_for_device = []

    for cpu_range_str in affinity_cpu_list:
        segments = [seg.strip() for seg in str(cpu_range_str).split(",") if seg.strip()]
        for segment in segments:
            if "-" in segment:
                endpoints = segment.split("-")
                if len(endpoints) != 2:
                    raise RuntimeError(
                        f"CPU segment '{segment}' in affinity_cpu_list:{affinity_cpu_list} is invalid."
                    )
                start_val, end_val = int(endpoints[0]), int(endpoints[1])
                for cid in range(start_val, end_val + 1):
                    if cid not in available_cpus:
                        raise RuntimeError(
                            f"CPU id:{cid} set in affinity_cpu_list:{affinity_cpu_list} is not available."
                        )
                    cpu_list_for_device.append(cid)
            else:
                cid = int(segment)
                if cid not in available_cpus:
                    raise RuntimeError(
                        f"CPU id:{cid} set in affinity_cpu_list:{affinity_cpu_list} is not available."
                    )
                cpu_list_for_device.append(cid)

    if not cpu_list_for_device:
        logger.warning("Available CPUs is less than 1. Will not enable bind core feature.")

    return cpu_list_for_device


def _assign_cpu_to_module(cpu_list_for_device, module_to_cpu_dict):
    """
    Assign specific CPUs to modules.

    Args:
        cpu_list_for_device (list): A map of device to its affinity CPUs.
        module_to_cpu_dict (dict): A map of module to its affinity CPU index in cpu_list_for_device.

    Returns:
        dict: Mapping of device to its affinity CPUs based on module segmentation.
    """
    module_bind_core_strategy = {}

    valid_module_names = {"main", "runtime", "pynative", "minddata"}

    if module_to_cpu_dict is not None:
        module_bind_core_strategy = {
            module: [cpu_list_for_device[i] for i in indices if 0 <= i < len(cpu_list_for_device)]
            for module, indices in module_to_cpu_dict.items() if module in valid_module_names
        }
    else:
        module_bind_core_strategy["main"] = cpu_list_for_device

    return module_bind_core_strategy


def _get_cpu_affinity_strategy(affinity_cpu_list=None, module_to_cpu_dict=None):
    """
    The entry to get bind-core strategy.

    Args:
        affinity_cpu_list (list, optional): User-configured CPU range to generate customized bind-core strategy.
          Default: ``None``.
        module_to_cpu_dict (dict, optional): User-configured module to CPU index to generate customized
          bind-core strategy. Default: ``None``.

    Returns:
        dict: Mapping of device to its affinity CPUs based on module segmentation.
    """
    device_target = context.get_context("device_target")
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()

    # Get the CPU resources in the environment. If this fails, the binding core feature will not be enabled.
    available_cpus = _get_available_cpus()
    if not available_cpus:
        logger.warning("Failed to acquire available cpu info from '/sys/fs/cgroup/cpuset/cpuset.cpus'. "
                       "Will not enable bind core feature.")
        return {}

    if (affinity_cpu_list is not None) and (affinity_cpu_list):
        # User configured bind-core strategy.
        cpu_list_for_device = _customize_generate_strategy(affinity_cpu_list, available_cpus)
    else:
        # Automatic generation of bind-core strategy based on resources on the environment.
        env_msrun_cpu_list = os.getenv("MSRUN_CPU_LIST")
        if env_msrun_cpu_list:
            module_bind_core_strategy = _assign_cpu_to_module(ast.literal_eval(env_msrun_cpu_list), module_to_cpu_dict)
            logger.warning(f"Module bind core policy from msrun: {module_bind_core_strategy}.")
            return module_bind_core_strategy
        try:
            logical_device_id = context.get_context("device_id")
            device_count = get_local_rank_size()
            physical_device_id = _get_physical_device_id(logical_device_id, simulation_level)
        except RuntimeError as e:
            logger.warning(f"Fail to get device_id or device_count, from {e} Will not enable bind core feature.")
            return {}
        # If the device target is Ascend, the affinity between the device and NUMA node is taken into account
        # to generate the binding core strategy.
        if device_target == "Ascend" and not simulation_level:
            device_to_cpu_map = _auto_generate_bind_core_strategy(device_count, available_cpus)
        else:
            device_to_cpu_map = _equal_distribute_cpu_strategy(device_count, available_cpus)
        # Get cpu_list for this process according to global device_to_cpu_map.
        cpu_list_for_device = device_to_cpu_map.get(physical_device_id, [])
    # cpu_list_for_device is empty, indicating that the basic conditions have not been met
    # to enable the thread bind core feature.
    if not cpu_list_for_device:
        return {}

    module_bind_core_strategy = _assign_cpu_to_module(cpu_list_for_device, module_to_cpu_dict)
    logger.warning(f"Module bind core policy generated: {module_bind_core_strategy}.")

    return module_bind_core_strategy
