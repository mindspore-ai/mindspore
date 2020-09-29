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


import os
import socket

RANK_TABLE_SAVE_PATH = './rank_table_8p.json'


def main():
    nproc_per_node = 4

    visible_devices = ['0', '1', '2', '3']

    server_id = socket.gethostbyname(socket.gethostname())

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
    hccn_table['board_id'] = '0x002f'  # A+K

    hccn_table['chip_info'] = '910'
    hccn_table['deploy_mode'] = 'lab'
    hccn_table['group_count'] = '1'
    hccn_table['group_list'] = []
    instance_list = []
    for instance_id in range(nproc_per_node):
        instance = {}
        instance['devices'] = []
        device_id = visible_devices[instance_id]
        device_ip = device_ips[device_id]
        instance['devices'].append({
            'device_id': device_id,
            'device_ip': device_ip,
        })
        instance['rank_id'] = str(instance_id)
        instance['server_id'] = server_id
        instance_list.append(instance)
    hccn_table['group_list'].append({
        'device_num': str(nproc_per_node),
        'server_num': '1',
        'group_name': '',
        'instance_count': str(nproc_per_node),
        'instance_list': instance_list,
    })
    hccn_table['para_plane_nic_location'] = 'device'
    hccn_table['para_plane_nic_name'] = []
    for instance_id in range(nproc_per_node):
        eth_id = visible_devices[instance_id]
        hccn_table['para_plane_nic_name'].append('eth{}'.format(eth_id))
    hccn_table['para_plane_nic_num'] = str(nproc_per_node)
    hccn_table['status'] = 'completed'
    import json
    with open(RANK_TABLE_SAVE_PATH, 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)


if __name__ == '__main__':
    if os.path.exists(RANK_TABLE_SAVE_PATH):
        print('Rank table file exists.')
    else:
        print('Generating rank table file.')
        main()
        print('Rank table file generated')
