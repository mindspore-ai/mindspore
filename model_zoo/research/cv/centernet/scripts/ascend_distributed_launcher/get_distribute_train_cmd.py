# Copyright 2020 Huawei Technologies Co., Ltd
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
"""distribute pretrain script"""
import os
import json
import configparser
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
    parser = ArgumentParser(description="mindspore distributed training")

    parser.add_argument("--run_script_dir", type=str, default="",
                        help="Run script path, it is better to use absolute path")
    parser.add_argument("--hyper_parameter_config_dir", type=str, default="",
                        help="Hyper Parameter config path, it is better to use absolute path")
    parser.add_argument("--mindrecord_dir", type=str, default="", help="Mindrecord dataset directory")
    parser.add_argument("--load_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--hccl_config_dir", type=str, default="",
                        help="Hccl config path, it is better to use absolute path")
    parser.add_argument("--cmd_file", type=str, default="distributed_cmd.sh",
                        help="Path of the generated cmd file.")
    parser.add_argument("--hccl_time_out", type=int, default=120,
                        help="Seconds to determine the hccl time out,"
                             "default: 120, which is the same as hccl default config")

    args = parser.parse_args()
    return args


def append_cmd(cmd, s):
    cmd += s
    cmd += "\n"
    return cmd

def append_cmd_env(cmd, key, value):
    return append_cmd(cmd, "export " + str(key) + "=" + str(value))

def distribute_train():
    """
    distribute pretrain scripts. The number of Ascend accelerators can be automatically allocated
    based on the device_num set in hccl config file, You don not need to specify that.
    """
    cmd = ""
    print("start", __file__)
    args = parse_args()

    run_script = args.run_script_dir
    mindrecord_dir = args.mindrecord_dir
    load_checkpoint_path = args.load_checkpoint_path
    cf = configparser.ConfigParser()
    cf.read(args.hyper_parameter_config_dir)
    cfg = dict(cf.items("config"))

    print("hccl_config_dir:", args.hccl_config_dir)
    print("hccl_time_out:", args.hccl_time_out)
    cmd = append_cmd_env(cmd, 'HCCL_CONNECT_TIMEOUT', args.hccl_time_out)
    cmd = append_cmd_env(cmd, 'RANK_TABLE_FILE', args.hccl_config_dir)

    cores = multiprocessing.cpu_count()
    print("the number of logical core:", cores)

    # get device_ips
    device_ips = {}
    with open('/etc/hccn.conf', 'r') as fin:
        for hccn_item in fin.readlines():
            if hccn_item.strip().startswith('address_'):
                device_id, device_ip = hccn_item.split('=')
                device_id = device_id.split('_')[1]
                device_ips[device_id] = device_ip.strip()

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
        device_id = instance["device_id"]
        rank_id = instance["rank_id"]
        print("\nstart training for rank " + str(rank_id) + ", device " + str(device_id) + ":")
        print("rank_id:", rank_id)
        print("device_id:", device_id)

        start = count * int(avg_core_per_rank)
        count += 1
        end = start + core_gap
        cmdopt = str(start) + "-" + str(end)

        cmd = append_cmd_env(cmd, "DEVICE_ID", str(device_id))
        cmd = append_cmd_env(cmd, "RANK_ID", str(rank_id))
        cmd = append_cmd_env(cmd, "DEPLOY_MODE", '0')
        cmd = append_cmd_env(cmd, "GE_USE_STATIC_MEMORY", '1')

        cmd = append_cmd(cmd, "rm -rf LOG" + str(device_id))
        cmd = append_cmd(cmd, "mkdir ./LOG" + str(device_id))
        cmd = append_cmd(cmd, "cp *.py ./LOG" + str(device_id))
        cmd = append_cmd(cmd, "mkdir -p ./LOG" + str(device_id) + "/ms_log")
        cmd = append_cmd(cmd, "env > ./LOG" + str(device_id) + "/env.log")

        cur_dir = os.getcwd()
        cmd = append_cmd_env(cmd, "GLOG_log_dir", cur_dir + "/LOG" + str(device_id) + "/ms_log")
        cmd = append_cmd_env(cmd, "GLOG_logtostderr", "0")

        print("core_nums:", cmdopt)
        print("epoch_size:", str(cfg['epoch_size']))
        print("mindrecord_dir:", mindrecord_dir)
        print("log_file_dir: " + cur_dir + "/LOG" + str(device_id) + "/training_log.txt")

        cmd = append_cmd(cmd, "cd " + cur_dir + "/LOG" + str(device_id))

        run_cmd = 'taskset -c ' + cmdopt + ' nohup python ' + run_script + " "
        opt = " ".join(["--" + key + "=" + str(cfg[key]) for key in cfg.keys()])
        if ('device_id' in opt) or ('device_num' in opt) or ('mindrecord_dir' in opt):
            raise ValueError("hyper_parameter_config.ini can not setting 'device_id',"
                             " 'device_num' or 'mindrecord_dir'! ")
        run_cmd += opt
        run_cmd += " --mindrecord_dir=" + mindrecord_dir
        run_cmd += " --load_checkpoint_path=" + load_checkpoint_path
        run_cmd += ' --device_id=' + str(device_id) + ' --device_num=' \
               + str(rank_size) + ' >./training_log.txt 2>&1 &'

        cmd = append_cmd(cmd, run_cmd)
        cmd = append_cmd(cmd, "cd -")
        cmd += "\n"

    with open(args.cmd_file, "w") as f:
        f.write(cmd)

if __name__ == "__main__":
    distribute_train()
