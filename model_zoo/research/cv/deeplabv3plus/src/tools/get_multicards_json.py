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
""""get_multicards_json"""
import os
import sys


def get_multicards_json(server_id):
    """ get_multicards_json"""
    hccn_configs = open('/etc/hccn.conf', 'r').readlines()
    device_ips = {}
    for hccn_item in hccn_configs:
        hccn_item = hccn_item.strip()
        if hccn_item.startswith('address_'):
            device_id, device_ip = hccn_item.split('=')
            device_id = device_id.split('_')[1]
            device_ips[device_id] = device_ip
            print('device_id:{}, device_ip:{}'.format(device_id, device_ip))
    hccn_table = {'board_id': '0x0000', 'chip_info': '910', 'deploy_mode': 'lab', 'group_count': '1', 'group_list': []}
    instance_list = []
    usable_dev = ''
    for instance_id in range(8):
        instance = {'devices': []}
        device_id = str(instance_id)
        device_ip = device_ips[device_id]
        usable_dev += str(device_id)
        instance['devices'].append({
            'device_id': device_id,
            'device_ip': device_ip,
        })
        instance['rank_id'] = str(instance_id)
        instance['server_id'] = server_id
        instance_list.append(instance)
    hccn_table['group_list'].append({
        'device_num': '8',
        'server_num': '1',
        'group_name': '',
        'instance_count': '8',
        'instance_list': instance_list,
    })
    hccn_table['para_plane_nic_location'] = 'device'
    hccn_table['para_plane_nic_name'] = []
    for instance_id in range(8):
        hccn_table['para_plane_nic_name'].append('eth{}'.format(instance_id))
    hccn_table['para_plane_nic_num'] = '8'
    hccn_table['status'] = 'completed'
    import json
    table_fn = os.path.join(os.getcwd(), 'rank_table_8p.json')
    print(table_fn)
    with open(table_fn, 'w') as table_fp:
        json.dump(hccn_table, table_fp, indent=4)


host_server_id = sys.argv[1]
get_multicards_json(host_server_id)
