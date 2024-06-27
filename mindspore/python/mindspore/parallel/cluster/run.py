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
from argparse import REMAINDER, ArgumentParser
from .process_entity import _ProcessManager

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
        help="specifies the IP address of the scheduler and its data type is string."
        " Allowed values: valid IP addresses."
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
        type=ast.literal_eval,
        choices=[True, False],
        help="specifies whether msrun should bind cpu cores to spawned processes."
    )
    parser.add_argument(
        "--sim_level",
        default=-1,
        type=int,
        choices=[0, 1],
        help="specifies simulation level. When this argument is set, msrun only spawns one process "
             "but export RANK_SIZE with value worker_num and RANK_ID with value sim_rank_id."
    )
    parser.add_argument(
        "--sim_rank_id",
        default=0,
        type=int,
        help="specifies simulation process's rank id. Only one process is spawned in simulation scenario."
    )
    parser.add_argument(
        "--rank_table_file",
        default="",
        type=str,
        help="specifies rank table file path. This path is not used to initialize distributed job in "
             "'rank table file manner' but to help support other features."
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
