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
    parser.add_argument("--data_dir", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--hccl_config_dir", type=str, default="",
                        help="Hccl config path, it is better to use absolute path")

    args = parser.parse_args()
    return args


def distribute_pretrain():
    """
    distribute pretrain scripts. The number of D chips can be automatically allocated
    based on the device_num set in hccl config file, You don not need to specify that.
    """
    print("start", __file__)
    args = parse_args()

    run_script = args.run_script_dir
    data_dir = args.data_dir
    cf = configparser.ConfigParser()
    cf.read(args.hyper_parameter_config_dir)
    cfg = dict(cf.items("config"))

    print("hccl_config_dir:", args.hccl_config_dir)
    os.environ['RANK_TABLE_FILE'] = args.hccl_config_dir

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

    os.environ['RANK_SIZE'] = str(rank_size)
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

        os.environ["DEVICE_ID"] = device_id
        os.environ["RANK_ID"] = rank_id
        os.environ["DEPLOY_MODE"] = "0"
        os.environ["GE_USE_STATIC_MEMORY"] = "1"

        os.system("rm -rf LOG" + str(device_id))
        os.system("mkdir ./LOG" + str(device_id))
        os.system("cp *.py ./LOG" + str(device_id))
        os.system("mkdir -p ./LOG" + str(device_id) + "/ms_log")
        os.system("env > ./LOG" + str(device_id) + "/env.log")

        cur_dir = os.getcwd()
        os.environ["GLOG_log_dir"] = cur_dir + "/LOG" + str(device_id) + "/ms_log"
        os.environ["GLOG_logtostderr"] = "0"

        print("core_nums:", cmdopt)
        print("epoch_size:", str(cfg['epoch_size']))
        print("data_dir:", data_dir)
        print("log_file_dir: ./LOG" + str(device_id) + "/log.txt")

        cmd = 'taskset -c ' + cmdopt + ' python ' + run_script + " "
        opt = " ".join(["--" + key + "=" + str(cfg[key]) for key in cfg.keys()])
        if ('device_id' in opt) or ('device_num' in opt) or ('data_dir' in opt):
            raise ValueError("hyper_parameter_config.ini can not setting 'device_id',"
                             " 'device_num' or 'data_dir'! ")
        cmd += opt
        cmd += " --data_dir=" + data_dir
        cmd += ' --device_id=' + str(device_id) + ' --device_num=' \
               + str(rank_size) + ' >./LOG' + str(device_id) + '/log.txt 2>&1 &'

        os.system(cmd)


if __name__ == "__main__":
    distribute_pretrain()
