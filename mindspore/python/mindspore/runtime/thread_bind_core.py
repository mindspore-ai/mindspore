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
import subprocess
import re
import os
import ast
from mindspore import log as logger
from mindspore import context
from mindspore.communication import get_local_rank_size


def execute_command(cmd_list, timeout=1000.0):
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

    range_pattern = re.compile(r'^\d+-\d+$')

    for cpu_range in affinity_cpu_list:
        if not isinstance(cpu_range, str):
            raise ValueError(f"CPU range '{cpu_range}' in '{affinity_cpu_list}' should be a string.")
        if not range_pattern.match(cpu_range):
            raise ValueError(f"CPU range '{cpu_range}' in '{affinity_cpu_list}' should be in format 'cpuidX-cpuidY'.")


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
        for cpu_id in cpu_indices:
            if not isinstance(cpu_id, int) or cpu_id < 0:
                raise ValueError(f"CPU index '{cpu_id}' for module '{module_name}' in '{cpu_indices}' "
                                 "should be a non-negative integer.")


def _get_cpu_available():
    """
    Get the CPU resources available on the environment.

    Returns:
        list: List of available CPUs on the environment.
    """
    available_cpus = []

    available_cpu_str = execute_command(["cat", "/sys/fs/cgroup/cpuset/cpuset.cpus"]).strip().split(",")
    if not available_cpu_str or (len(available_cpu_str) == 1 and not available_cpu_str[0]):
        raise RuntimeError("Empty available CPU range in '/sys/fs/cgroup/cpuset/cpuset.cpus'.")
    for range_str in available_cpu_str:
        endpoints = range_str.strip().split("-")
        if len(endpoints) == 1:
            available_cpus.append(int(endpoints[0]))
        elif len(endpoints) == 2:
            start = int(endpoints[0])
            end = int(endpoints[1])
            if start > end:
                raise RuntimeError(f"Invalid CPU range: {range_str} in '/sys/fs/cgroup/cpuset/cpuset.cpus'.")
            available_cpus.extend(range(start, end + 1))
        else:
            raise RuntimeError("Failed to parse the result of executing 'cat /sys/fs/cgroup/cpuset/cpuset.cpus'.")

    return sorted(available_cpus)


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

    return device_map_info, available_devices


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
    numa_to_device_map[-1] = list(device_to_pcie_map.keys())

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


def _equal_distribution_strategy(device_count, available_cpus):
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
        for device_rank, device_id in enumerate(available_devices):
            cpu_start = device_rank * cpu_num_per_device
            device_to_cpu_map[device_id] = available_cpus[cpu_start: cpu_start + cpu_num_per_device]

    return device_to_cpu_map


def _auto_generate_strategy(device_count, available_cpus):
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
        device_to_cpu_map = _equal_distribution_strategy(device_count, available_cpus)
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
        endpoints = cpu_range_str.split("-")
        for cid in range(int(endpoints[0]), int(endpoints[1]) + 1):
            if cid not in available_cpus:
                raise RuntimeError(f"CPU id:{cid} set in affinity_cpu_list:{affinity_cpu_list} is not available.")
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
    try:
        available_cpus = _get_cpu_available()
    except RuntimeError as e:
        logger.warning(f"Failed to acquire available cpu info, from {e} Will not enable bind core feature.")
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
            device_to_cpu_map = _auto_generate_strategy(device_count, available_cpus)
        else:
            device_to_cpu_map = _equal_distribution_strategy(device_count, available_cpus)
        # Get cpu_list for this process according to global device_to_cpu_map.
        cpu_list_for_device = device_to_cpu_map.get(physical_device_id, [])
    # cpu_list_for_device is empty, indicating that the basic conditions have not been met
    # to enable the thread bind core feature.
    if not cpu_list_for_device:
        return {}

    module_bind_core_strategy = _assign_cpu_to_module(cpu_list_for_device, module_to_cpu_dict)
    logger.warning(f"Module bind core policy generated: {module_bind_core_strategy}.")

    return module_bind_core_strategy
