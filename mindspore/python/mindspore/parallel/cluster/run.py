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
"""Entrypoint of ms_run"""
import ast
import re
import json
import os
import hashlib
from argparse import REMAINDER, ArgumentParser, ArgumentTypeError
from .process_entity import _ProcessManager


def _parse_and_validate_bind_generic(value, resource):
    """Generic parsing and validation function for bind parameters."""
    if resource == "CPU":
        desc = "cpuidX"
    elif resource == "NUMA":
        desc = "numaidX"
    else:
        raise ValueError(f"Unsupported bind {resource}.")

    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        try:
            value_dict = json.loads(value)
        except json.JSONDecodeError as e:
            raise ArgumentTypeError("Failed to parse JSON into a dictionary") from e
    elif isinstance(value, dict):
        value_dict = value
    else:
        raise ArgumentTypeError(f"Type of {value} should be bool or dict, but got {type(value)}.")

    if isinstance(value_dict, dict):
        # New-style bind file uses dedicated keys, skip per-device validation here.
        if any(k in value_dict for k in ("bind_config", "bind_cpu", "bind_memory")):
            return value_dict
        resource_pattern = re.compile(r'^(\d+-\d+|\d+(?:,\d+)*)$')

        for device, affinity_list in value_dict.items():
            if not re.fullmatch(r"device\d+|scheduler", device):
                raise ArgumentTypeError(f"Key {device} must be in format 'scheduler' or 'deviceX'.")

            if not isinstance(affinity_list, list):
                raise ArgumentTypeError(
                    f"Type of value for {device}:{affinity_list} should be a list, but got {type(affinity_list)}."
                )

            normalized_list = []
            for item in affinity_list:
                if not isinstance(item, str):
                    raise ArgumentTypeError(
                        f"{resource} {item} in {affinity_list} should be a string."
                    )
                cleaned_item = re.sub(r'\s+', '', item)
                if not resource_pattern.match(cleaned_item):
                    raise ArgumentTypeError(
                        f"{resource} {item} in {affinity_list} should be in format "
                        f"'{desc}', '{desc},{desc[0:-1]}Y' or '{desc}-{desc[0:-1]}Y'."
                    )
                normalized_list.append(cleaned_item)
            value_dict[device] = normalized_list
        return value_dict

    raise ArgumentTypeError(f"Type of {value} should be bool or dict, but got {type(value_dict)}.")


def _parse_and_validate_bind_core(value):
    """Parse input argument of --bind_core."""
    return _parse_and_validate_bind_generic(value, "CPU")


def _parse_and_validate_bind_numa(value):
    """Parse input argument of --bind_numa."""
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            return _parse_and_validate_bind_generic(value, "NUMA")
        if value.lower().endswith(".json") and not os.path.exists(value):
            raise ArgumentTypeError(f"bind_numa json file '{value}' does not exist.")
        if value.lower().endswith(".json") and not os.path.isfile(value):
            raise ArgumentTypeError(f"bind_numa json path '{value}' is not a file.")
    if isinstance(value, str) and os.path.isfile(value):
        abs_path = os.path.abspath(value)
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                raw = f.read()
                config = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ArgumentTypeError(f"Failed to load bind_numa json file: {value}.") from exc
        if not isinstance(config, dict):
            raise ArgumentTypeError(f"bind_numa json file '{value}' should be a json object.")
        bind_config = config.get("bind_config", {})
        if not isinstance(bind_config, dict):
            raise ArgumentTypeError("bind_config must be a json object.")
        cpu_mode = str(bind_config.get("bind_cpu_mode", "none")).lower()
        if cpu_mode not in ("cpu", "numa", "none"):
            raise ArgumentTypeError("'bind_cpu_mode' only supports 'cpu'/'numa'/'none'.")
        mem_mode = str(bind_config.get("bind_memory_mode", "none")).lower()
        if mem_mode not in ("numa", "none"):
            raise ArgumentTypeError("'bind_memory_mode' only supports 'numa'/'none'.")
        os.environ["MSRUN_BIND_FILE"] = abs_path
        os.environ["MSRUN_BIND_FILE_HASH"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return config
    return _parse_and_validate_bind_generic(value, "NUMA")


def get_args():
    """
    Parses and retrieves command-line arguments.

    """
    parser = ArgumentParser()
    parser.add_argument(
        "--worker_num", type=int, default=8,
        help="the total number of nodes participating in the training, an integer variable, "
        "with a default value of 8."
    )
    parser.add_argument(
        "--local_worker_num",
        type=int, default=8,
        help="the number of nodes participating in local training, an integer variable, "
        "with a default value of 8."
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1", type=str,
        help="specifies the IP address or the host name of the scheduler and its data type is string."
        " Allowed values: valid IP addresses or valid host name."
    )
    parser.add_argument(
        "--master_port", default=8118, type=int,
        help="specifies the port number of the scheduler, and its data type is integer."
        " Allowed values: port numbers within the range of 1024 to 65535 that are not "
        "already in use."
    )
    parser.add_argument(
        "--node_rank", default=-1, type=int,
        help="specifies the rank of current physical node, and its data type is integer."
        " This parameter is used for rank id assignment for each process on the node."
        " If not set, MindSpore will assign rank ids automatically and"
        " rank id of each process on the same node will be continuous."
    )
    parser.add_argument(
        "--log_dir", default="", type=str,
        help="specifies the log output file path."
    )
    parser.add_argument(
        "--join",
        default=False,
        type=ast.literal_eval,
        choices=[True, False],
        help="specifies whether msrun should join spawned processes and return distributed job results."
             "If set to True, msrun will check process status and parse the log files."
    )
    parser.add_argument(
        "--cluster_time_out",
        default=600,
        type=int,
        help="specifies time out window of cluster building procedure in second. "
             "If only scheduler is launched, or spawned worker number is not enough, "
             "other processes will wait for 'cluster_time_out' seconds and then exit. "
             "If this value is negative, other processes will wait infinitely."
    )
    parser.add_argument(
        "--bind_core",
        default=False,
        type=_parse_and_validate_bind_core,
        help="specifies whether msrun should bind CPU cores to spawned processes. "
             "If set to True, msrun will bind core based on the environment automatically, "
             "and if passed a dict, msrun will bind core based on this dict information."
    )
    parser.add_argument(
        "--bind_numa",
        default=False,
        type=_parse_and_validate_bind_numa,
        help="specifies whether msrun should bind NUMA to spawned processes. "
             "If set to True, msrun will bind NUMA based on the environment automatically, "
             "and if passed a dict, msrun will bind NUMA based on this dict information."
    )
    parser.add_argument(
        "--sim_level",
        default=-1,
        type=int,
        choices=[0, 1, 2, 3],
        help="specifies simulation level. This argument activates dryrun mode, functioning "
             "equivalently to environment variable 'MS_SIMULATION_LEVEL' while having higher priority."
    )
    parser.add_argument(
        "--sim_rank_id",
        default=-1,
        type=int,
        help="specifies simulation process's rank id. When this argument is set, only one process "
             "is spawned on dryrun mode, functioning equivalently to environment variable 'RANK_ID' "
             "while having higher priority."
    )
    parser.add_argument(
        "--rank_table_file",
        default="",
        type=str,
        help="specifies rank table file path. This path is not used to initialize distributed job in "
             "'rank table file manner' but to help support other features."
    )
    parser.add_argument(
        "--worker_log_name",
        default="",
        type=str,
        help="Specifies the worker log file name as a string for current node; the default is worker_[rankid]. "
             "Support configuring the current IP address and host name by using {ip} and {hostname} respectively. "
             "e.g. --worker_log_name=worker_{ip}_{hostname}_test, worker [rankid] log name for current node "
             "will be worker_[real IP address]_[real host name]_test_[rankid]."
    )
    parser.add_argument(
        "--tail_worker_log",
        default="-1",
        type=str,
        help="Only tail worker log to console when '--join=True' and the configured value should be within "
             "[0, local_worker_num], otherwise worker log will not be tail. All worker logs will be tail by "
             "default. Support tail the specified worker log (e.g. --tail_log=0 tail the worker 0 log to console)."
    )
    parser.add_argument(
        "task_script",
        type=str,
        help="The full path to the script that will be launched in distributed manner, followed "
             "by any additional arguments required by the script."
    )
    parser.add_argument(
        "task_script_args", nargs=REMAINDER,
        help="Arguments for user-defined script."
    )
    return parser.parse_args()


def run(args):
    """
    Runs the dynamic networking process manager.

    Args:
        args: An object containing the command-line arguments.

    """
    process_manager = _ProcessManager(args)
    process_manager.run()


def main():
    """the main function"""
    args = get_args()
    run(args)

if __name__ == "__main__":
    main()
