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
"""Binding helpers for ms_run."""
import os
import subprocess
import mindspore.log as logger
from mindspore.runtime.thread_bind_core import (
    _get_cpu_info,
    _get_physical_device_id,
    _get_available_cpus,
    _get_available_numas,
    _expand_ranges,
    _auto_generate_bind_core_strategy,
    _auto_generate_bind_numa_strategy,
    _equal_distribute_cpu_strategy,
    _equal_distribute_numa_strategy,
)


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


def _is_new_bind_config(bind_cfg):
    """
    Check whether bind_cfg follows the unified bind json schema.
    """
    return isinstance(bind_cfg, dict) and "bind_config" in bind_cfg


def _normalize_cpu_binding(cpu_value):
    """
    Normalize CPU binding configuration to a string acceptable by taskset/numactl.
    Supports str.
    """
    if not cpu_value:
        return ""
    if isinstance(cpu_value, str):
        return cpu_value.replace(" ", "")
    raise ValueError(f"bind_cpu main should be str, but got {type(cpu_value)}")


def _get_bind_device_key(local_rank_id):
    """
    Get device key for bind file based on physical device id.
    """
    if local_rank_id == -1:
        return "scheduler"
    simulation_level = os.getenv("MS_SIMULATION_LEVEL", "").strip()
    try:
        physical_device_id = _get_physical_device_id(local_rank_id, simulation_level)
    except RuntimeError as e:
        logger.warning(f"Failed to acquire device id for rank {local_rank_id}, from {e}")
        return None
    return f"device{physical_device_id}"


def _get_bind_modes_and_maps(bind_cfg):
    """
    Extract bind modes and related maps from unified bind json configuration.
    """
    bind_config = bind_cfg.get("bind_config", {})
    bind_cpu_mode = str(bind_config.get("bind_cpu_mode", "none")).lower()
    bind_memory_mode = str(bind_config.get("bind_memory_mode", "none")).lower()
    bind_cpu_map = bind_cfg.get("bind_cpu", {})
    bind_memory_map = bind_cfg.get("bind_memory", {})
    return bind_cpu_mode, bind_memory_mode, bind_cpu_map, bind_memory_map


def _get_device_cpu_binding(device_key, bind_cpu_mode, bind_cpu_map):
    """
    Get CPU binding values (main_cpu or cpu_node) for a device.
    """
    if bind_cpu_mode == "cpu":
        device_cpu_cfg = bind_cpu_map.get(device_key)
        if device_cpu_cfg is None:
            device_cpu_cfg = {}
        return _normalize_cpu_binding(device_cpu_cfg.get("main")), None
    if bind_cpu_mode == "numa":
        cpu_node_val = bind_cpu_map.get(device_key)
        return None, str(cpu_node_val) if cpu_node_val is not None else None
    return None, None


def _get_device_memory_binding(device_key, bind_memory_mode, bind_memory_map):
    """
    Get memory binding value for a device.
    """
    if bind_memory_mode != "numa":
        return None
    mem_val = bind_memory_map.get(device_key)
    return str(mem_val) if mem_val is not None else None


def _build_bind_prefix_for_cpu_mode(bind_memory_mode, main_cpu, memory_node):
    """
    Build bind prefix when bind_cpu_mode is 'cpu'.
    """
    if bind_memory_mode == "numa" and memory_node is not None:
        prefix = ["numactl", "--membind", memory_node]
        if main_cpu:
            prefix.extend(["--physcpubind", main_cpu])
        return prefix
    if main_cpu:
        return ["taskset", "-c", main_cpu]
    return []


def _build_bind_prefix_for_numa_mode(bind_memory_mode, cpu_node, memory_node):
    """
    Build bind prefix when bind_cpu_mode is 'numa'.
    """
    if cpu_node is None:
        if bind_memory_mode == "numa" and memory_node is not None:
            return ["numactl", "--membind", memory_node]
        return []
    prefix = ["numactl"]
    if bind_memory_mode == "numa" and memory_node is not None:
        prefix.extend(["--membind", memory_node])
    prefix.extend(["--cpunodebind", cpu_node])
    return prefix


def _build_bind_prefix_for_no_cpu_mode(bind_memory_mode, memory_node):
    """
    Build bind prefix when bind_cpu_mode is not set.
    """
    if bind_memory_mode == "numa" and memory_node is not None:
        return ["numactl", "--membind", memory_node]
    return []


def _generate_bind_prefix_from_config(local_rank, bind_cfg):
    """
    Generate binding prefix according to unified bind json configuration.
    """
    device_key = _get_bind_device_key(local_rank)
    if device_key is None:
        return []

    bind_cpu_mode, bind_memory_mode, bind_cpu_map, bind_memory_map = _get_bind_modes_and_maps(bind_cfg)
    main_cpu, cpu_node = _get_device_cpu_binding(device_key, bind_cpu_mode, bind_cpu_map)
    memory_node = _get_device_memory_binding(device_key, bind_memory_mode, bind_memory_map)

    if bind_cpu_mode == "cpu":
        return _build_bind_prefix_for_cpu_mode(bind_memory_mode, main_cpu, memory_node)
    if bind_cpu_mode == "numa":
        return _build_bind_prefix_for_numa_mode(bind_memory_mode, cpu_node, memory_node)
    return _build_bind_prefix_for_no_cpu_mode(bind_memory_mode, memory_node)


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


def _validate_bind_cpu_mode_cpu(bind_cpu, available_cpus):
    """
    Validate bind_cpu when bind_cpu_mode is 'cpu'.
    """
    if not isinstance(bind_cpu, dict):
        raise TypeError("bind_cpu must be a dict when bind_cpu_mode is 'cpu'.")
    for device, device_cpu_cfg in bind_cpu.items():
        if device_cpu_cfg is None:
            continue
        if not isinstance(device_cpu_cfg, dict):
            raise TypeError(
                f"'bind_cpu_mode' is 'cpu', in 'bind_cpu', type of value for {device}:{device_cpu_cfg} "
                f"should be dict, but got {type(device_cpu_cfg)}."
            )
        for ranges in device_cpu_cfg.values():
            cpu_ids = _expand_ranges(ranges)
            for cpu_id in cpu_ids:
                if cpu_id not in available_cpus:
                    raise RuntimeError(f"CPU id {cpu_id} in {device_cpu_cfg} for {device} is not available.")


def _validate_bind_cpu_mode_numa(bind_cpu, available_numas):
    """
    Validate bind_cpu when bind_cpu_mode is 'numa'.
    """
    if not isinstance(bind_cpu, dict):
        raise TypeError("bind_cpu must be a dict when bind_cpu_mode is 'numa'.")
    for device, cpu_node_val in bind_cpu.items():
        if cpu_node_val is None:
            continue
        if not isinstance(cpu_node_val, (int, str)):
            raise TypeError(
                f"bind_cpu_mode is 'numa', in 'bind_cpu', type of value for {device}:{cpu_node_val} "
                f"should be int/str, but got {type(cpu_node_val)}."
            )
        numa_ids = _expand_ranges(cpu_node_val)
        for numa_id in numa_ids:
            if numa_id not in available_numas:
                raise RuntimeError(f"NUMA id {numa_id} in {cpu_node_val} for {device} is not available.")


def _validate_bind_memory_mode_numa(bind_memory, available_numas):
    """
    Validate bind_memory when bind_memory_mode is 'numa'.
    """
    if not isinstance(bind_memory, dict):
        raise TypeError("bind_memory must be a dict when bind_memory_mode is 'numa'.")
    for device, mem_node_val in bind_memory.items():
        if mem_node_val is None:
            continue
        if not isinstance(mem_node_val, (int, str)):
            raise TypeError(
                f"bind_memory_mode is 'numa', in 'bind_memory', type of value for {device}:{mem_node_val} "
                f"should be int/str, but got {type(mem_node_val)}."
            )
        numa_ids = _expand_ranges(mem_node_val)
        for numa_id in numa_ids:
            if numa_id not in available_numas:
                raise RuntimeError(f"NUMA id {numa_id} in {mem_node_val} for {device} is not available.")


def _validate_bind_config_availability(bind_cfg, available_numas, available_cpus):
    """
    Validate the availability of CPU/NUMA node IDs in unified bind config.
    """
    bind_cpu_mode, bind_memory_mode, bind_cpu, bind_memory = _get_bind_modes_and_maps(bind_cfg)

    if bind_cpu_mode == "cpu":
        _validate_bind_cpu_mode_cpu(bind_cpu, available_cpus)
    elif bind_cpu_mode == "numa":
        _validate_bind_cpu_mode_numa(bind_cpu, available_numas)

    if bind_memory_mode == "numa":
        _validate_bind_memory_mode_numa(bind_memory, available_numas)


def _generate_bind_context(local_worker_num, arg_bind_numa, arg_bind_core):
    """
    1. Validate the availability of mandatory system commands required for CPU core binding and memory binding
       in the current environment.
    2. Automatically generate the CPU/memory affinity policy when the auto-binding mode for CPU/memory is activated.
    3. Perform validity and format checks on the custom binding policy if it is provided as an input parameter.
    """
    device_to_numa_map = {}
    device_to_cpu_map = {}
    bind_context = BindContext()

    available_numas = _get_available_numas()
    available_cpus = _get_available_cpus()
    bind_context.available_numas = available_numas
    bind_context.available_cpus = available_cpus

    # If a JSON file is passed to the `--bind_numa` parameter, validate resource ids only.
    if _is_new_bind_config(arg_bind_numa):
        if not _check_command_availability("taskset") or not _check_command_availability("numactl"):
            raise RuntimeError("Command 'numactl'/'taskset' is not supported.")
        if not available_cpus or not available_numas:
            raise RuntimeError("No available CPU/NUMA in the environment.")
        _validate_bind_config_availability(arg_bind_numa, available_numas, available_cpus)
        bind_context.bind_numa = arg_bind_numa
        return bind_context

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

    bind_context.bind_numa = arg_bind_numa
    bind_context.bind_core = arg_bind_core
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
            logger.warning(f"No CPU assigned to device {physical_device_id}.")
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
            logger.warning(f"No NUMA assigned to device {physical_device_id}.")
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
