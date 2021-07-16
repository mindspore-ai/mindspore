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
"""launch train script"""
import os
import sys
import json
import subprocess
import shutil
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
    parser = ArgumentParser(description="mindspore distributed training launch "
                                        "helper utility that will spawn up "
                                        "multiple distributed processes")
    parser.add_argument("--nproc_per_node", type=int, default=1,
                        help="The number of processes to launch on each node, "
                             "for D training, this is recommended to be set "
                             "to the number of D in your system so that "
                             "each process can be bound to a single D.")
    parser.add_argument("--visible_devices", type=str, default="0,1,2,3,4,5,6,7",
                        help="will use the visible devices sequentially")
    parser.add_argument("--server_id", type=str, default="",
                        help="server ip")
    parser.add_argument("--training_script", type=str,
                        help="The full path to the single D training "
                             "program/script to be launched in parallel, "
                             "followed by all the arguments for the "
                             "training script")
    # rest from the training program
    args, unknown = parser.parse_known_args()
    args.training_script_args = unknown
    return args


def main():
    print("start", __file__)
    args = parse_args()
    print(args)
    visible_devices = args.visible_devices.split(',')
    assert os.path.isfile(args.training_script)
    assert len(visible_devices) >= args.nproc_per_node
    print('visible_devices:{}'.format(visible_devices))
    if not args.server_id:
        print('pleaser input server ip!!!')
        exit(0)
    print('server_id:{}'.format(args.server_id))

    # construct hccn_table
    hccn_configs = open('/etc/hccn.conf', 'r').readlines()
    device_ips = {}
    for hccn_item in hccn_configs:
        hccn_item = hccn_item.strip()
        if hccn_item.startswith('address_'):
            device_id, device_ip = hccn_item.split('=')
            device_id = device_id.split('_')[1]
            device_ips[device_id] = device_ip
            print('device_id:{}, device_ip:{}'.format(device_id, device_ip))
    hccn_table = {}
    hccn_table['board_id'] = '0x0000'
    hccn_table['chip_info'] = '910'
    hccn_table['deploy_mode'] = 'lab'
    hccn_table['group_count'] = '1'
    hccn_table['group_list'] = []
    instance_list = []
    usable_dev = ''
    for instance_id in range(args.nproc_per_node):
        instance = {}
        instance['devices'] = []
        device_id = visible_devices[instance_id]
        device_ip = device_ips[device_id]
        usable_dev += str(device_id)
        instance['devices'].append({
            'device_id': device_id,
            'device_ip': device_ip,
        })
        instance['rank_id'] = str(instance_id)
        instance['server_id'] = args.server_id
        instance_list.append(instance)
    hccn_table['group_list'].append({
        'device_num': str(args.nproc_per_node),
        'server_num': '1',
        'group_name': '',
        'instance_count': str(args.nproc_per_node),
        'instance_list': instance_list,
    })
    hccn_table['para_plane_nic_location'] = 'device'
    hccn_table['para_plane_nic_name'] = []
    for instance_id in range(args.nproc_per_node):
        eth_id = visible_devices[instance_id]
        hccn_table['para_plane_nic_name'].append('eth{}'.format(eth_id))
    hccn_table['para_plane_nic_num'] = str(args.nproc_per_node)
    hccn_table['status'] = 'completed'

    # save hccn_table to file
    table_path = os.getcwd()
    if not os.path.exists(table_path):
        os.mkdir(table_path)
    table_fn = os.path.join(table_path,
                            'rank_table_{}p_{}_{}.json'.format(args.nproc_per_node, usable_dev, args.server_id))
    with open(table_fn, 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)
    sys.stdout.flush()

    # spawn the processes
    processes = []
    cmds = []
    log_files = []
    env = os.environ.copy()
    env['RANK_SIZE'] = str(args.nproc_per_node)
    cur_path = os.getcwd()
    for rank_id in range(0, args.nproc_per_node):
        os.chdir(cur_path)
        device_id = visible_devices[rank_id]
        device_dir = os.path.join(cur_path, 'device{}'.format(rank_id))
        env['RANK_ID'] = str(rank_id)
        env['DEVICE_ID'] = str(device_id)
        if args.nproc_per_node > 1:
            env['RANK_TABLE_FILE'] = table_fn
        if os.path.exists(device_dir):
            shutil.rmtree(device_dir)
        os.mkdir(device_dir)
        os.chdir(device_dir)
        cmd = [sys.executable, '-u']
        cmd.append(args.training_script)
        cmd.extend(args.training_script_args)
        log_file = open(
            '{dir}/log{id}.log'.format(dir=device_dir, id=rank_id), 'w')
        process = subprocess.Popen(
            cmd, stdout=log_file, stderr=log_file, env=env)
        processes.append(process)
        cmds.append(cmd)
        log_files.append(log_file)
    for process, cmd, log_file in zip(processes, cmds, log_files):
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process, cmd=cmd)
        log_file.close()


if __name__ == "__main__":
    main()
