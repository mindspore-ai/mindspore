# Copyright 2021 Huawei Technologies Co., Ltd
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
"""distribute running script"""
import os
import json
import multiprocessing
from argparse import ArgumentParser


def parse_args():
    """
    parse args .

    Args:

    Returns:
        args.

    Examples:
        >>> parse_args()
    """
    parser = ArgumentParser(description="Distributed training scripts generator for MindSpore")

    parser.add_argument("--run_script_path", type=str, default="",
                        help="Run script path, it is better to use absolute path")
    parser.add_argument("--args", type=str, default="",
                        help="Other arguments which will be passed to main program directly")
    parser.add_argument("--hccl_config_dir", type=str, default="",
                        help="Hccl config path, it is better to use absolute path")
    parser.add_argument("--cmd_file", type=str, default="distributed_cmd.sh",
                        help="Path of the generated cmd file.")
    parser.add_argument("--hccl_time_out", type=int, default=120,
                        help="Seconds to determine the hccl time out,"
                             "default: 120, which is the same as hccl default config")
    parser.add_argument("--cpu_bind", action="store_true", default=False,
                        help="Bind cpu cores or not")

    args = parser.parse_args()
    return args


def append_cmd(cmd, s):
    cmd += s
    cmd += "\n"
    return cmd


def append_cmd_env(cmd, key, value):
    return append_cmd(cmd, "export " + str(key) + "=" + str(value))


def set_envs(cmd, logic_id, rank_id):
    """
    Set environment variables.
    """
    cmd = append_cmd_env(cmd, "DEVICE_ID", str(logic_id))
    cmd = append_cmd_env(cmd, "RANK_ID", str(rank_id))
    return cmd


def make_dirs(cmd, logic_id):
    """
    Make directories and change path.
    """
    cmd = append_cmd(cmd, "rm -rf LOG" + str(logic_id))
    cmd = append_cmd(cmd, "mkdir ./LOG" + str(logic_id))
    cmd = append_cmd(cmd, "mkdir -p ./LOG" + str(logic_id) + "/ms_log")
    cmd = append_cmd(cmd, "env > ./LOG" + str(logic_id) + "/env.log")
    cur_dir = os.getcwd()
    cmd = append_cmd_env(cmd, "GLOG_log_dir", cur_dir + "/LOG" + str(logic_id) + "/ms_log")
    cmd = append_cmd_env(cmd, "GLOG_logtostderr", "0")
    cmd = append_cmd(cmd, "cd " + cur_dir + "/LOG" + str(logic_id))
    return cmd


def print_info(rank_id, device_id, logic_id, cmdopt, cur_dir):
    """
    Print some information about scripts.
    """
    print("\nstart training for rank " + str(rank_id) + ", device " + str(device_id) + ":")
    print("rank_id:", rank_id)
    print("device_id:", device_id)
    print("logic_id", logic_id)
    print("core_nums:", cmdopt)
    print("log_file_dir: " + cur_dir + "/LOG" + str(logic_id) + "/pretraining_log.txt")

def distribute_run():
    """
    distribute pretrain scripts. The number of Ascend accelerators can be automatically allocated
    based on the device_num set in hccl config file, You don not need to specify that.
    """
    cmd = ""
    print("start", __file__)
    args = parse_args()

    run_script = args.run_script_path

    print("hccl_config_dir:", args.hccl_config_dir)
    print("hccl_time_out:", args.hccl_time_out)
    cmd = append_cmd_env(cmd, 'HCCL_CONNECT_TIMEOUT', args.hccl_time_out)
    cmd = append_cmd_env(cmd, 'RANK_TABLE_FILE', args.hccl_config_dir)

    cores = multiprocessing.cpu_count()
    print("the number of logical core:", cores)

    # get device_ips
    device_ips = {}
    physic_logic_ids = {}
    with open('/etc/hccn.conf', 'r') as fin:
        for hccn_item in fin.readlines():
            if hccn_item.strip().startswith('address_'):
                device_id, device_ip = hccn_item.split('=')
                device_id = device_id.split('_')[1]
                device_ips[device_id] = device_ip.strip()

    if not device_ips:
        raise ValueError("There is no address in /etc/hccn.conf")

    for logic_id, device_id in enumerate(sorted(device_ips.keys())):
        physic_logic_ids[device_id] = logic_id

    with open(args.hccl_config_dir, "r", encoding="utf-8") as fin:
        hccl_config = json.loads(fin.read())
        rank_size = 0
        for server in hccl_config["server_list"]:
            rank_size += len(server["device"])
            if server["device"][0]["device_ip"] in device_ips.values():
                this_server = server

    cmd = append_cmd_env(cmd, "RANK_SIZE", str(rank_size))
    print("total rank size:", rank_size)
    print("this server rank size:", len(this_server["device"]))
    avg_core_per_rank = int(int(cores) / len(this_server["device"]))
    core_gap = avg_core_per_rank - 1
    print("avg_core_per_rank:", avg_core_per_rank)

    count = 0
    for instance in this_server["device"]:
        # device_id is the physical id, we use logic id to specific the selected device.
        # While running on a server with 8 pcs, the logic ids are equal to the device ids.
        device_id = instance["device_id"]
        rank_id = instance["rank_id"]
        logic_id = physic_logic_ids[device_id]
        start = count * int(avg_core_per_rank)
        count += 1
        end = start + core_gap
        cmdopt = str(start) + "-" + str(end)
        cur_dir = os.getcwd()

        cmd = set_envs(cmd, logic_id, rank_id)
        cmd = make_dirs(cmd, logic_id)

        print_info(rank_id=rank_id, device_id=device_id, logic_id=logic_id, cmdopt=cmdopt, cur_dir=cur_dir)

        if args.cpu_bind:
            run_cmd = 'taskset -c ' + cmdopt + ' '
        else:
            run_cmd = ""
        run_cmd += 'nohup python ' + run_script + " "

        run_cmd += " " + ' '.join([str(x) for x in args.args.split(' ')[1:]])
        run_cmd += ' >./log.txt 2>&1 &'

        cmd = append_cmd(cmd, run_cmd)
        cmd = append_cmd(cmd, "cd -")
        cmd = append_cmd(cmd, "echo \"run with" +
                         " rank_id=" + str(rank_id) +
                         " device_id=" + str(device_id) +
                         " logic_id=" + str(logic_id) + "\"")
        cmd += "\n"

    with open(args.cmd_file, "w") as f:
        f.write(cmd)

if __name__ == "__main__":
    distribute_run()
