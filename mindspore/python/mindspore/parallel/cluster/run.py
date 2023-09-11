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
from argparse import REMAINDER, ArgumentParser
from process_entity import  _ProcessManager

def get_args():
    """
    Parses and retrieves command-line arguments.

    """
    parser = ArgumentParser()
    parser.add_argument(
        "--total_nodes",
        type=int,
        default=1,
        help="the total number of nodes participating in the training, an integer variable, "
        "with a default value of 1."
    )
    parser.add_argument(
        "--local_nodes",
        type=int,
        default=1,
        help="the number of nodes participating in local training, an integer variable, "
        "with a default value of 1."
    )
    parser.add_argument(
        "--master_addr",
        default="127.0.0.1",
        type=str,
        help="specifies the IP address of the scheduler and its data type is string."
        " Allowed values: valid IP addresses."
    )
    parser.add_argument(
        "--master_port",
        default=8118,
        type=int,
        help="specifies the port number of the scheduler, and its data type is integer."
        " Allowed values: port numbers within the range of 1024 to 65535 that are not "
        "already in use."
    )
    parser.add_argument(
        "--is_scalein",
        default=0,
        type=int,
        help="an integer parameter indicating if the task involves scaling in. It accepts "
        "values of 0 or 1, where 1 indicates the removal of `scale_num` nodes from "
        "the local cluster."
    )
    parser.add_argument(
        "--is_scaleout",
        default=0,
        type=int,
        help="an integer parameter indicating if the task involves scaling out. It accepts"
        " values of 0 or 1, where 1 indicates the addition of `scale_num` nodes to the"
        " local cluster."
    )
    parser.add_argument(
        "--scale_num",
        default=0,
        type=int,
        help=" specifies the number of nodes to be added or removed from the local cluster. "
    )
    parser.add_argument(
        "training_script",
        type=str,
        help="The full path to the training script that will be executed in parallel, followed "
        "by any additional arguments required by the script."
    )
    parser.add_argument(
        "training_script_args", nargs=REMAINDER
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
