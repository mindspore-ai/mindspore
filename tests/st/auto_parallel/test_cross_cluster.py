# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test cross cluster"""
import os
import subprocess


def _get_server_ip():
    cmd = "ifconfig -a | grep inet | grep -v 127.0.0.1 | grep -v inet6 | awk '{print $2}' | tr -d \"addr:\" | tail -n 1"
    server_ip = subprocess.getoutput(cmd)
    return server_ip


def _get_device_ips():
    device_ips = []
    result = subprocess.getoutput("for i in {0..7};do hccn_tool -i $i -ip -g;done")
    lines = result.splitlines()
    for line in lines:
        key, value = line.split(':')
        if key == 'ipaddr':
            device_ips.append(value)
    return device_ips


def set_cross_cluster_rank_table(rank_size=8, rank_table="cross_cluster_rank_table.json"):
    # get the server_ip in context.
    server_ip = _get_server_ip()
    # get a list of real device ips in context.
    device_ips = _get_device_ips()
    ret = os.system(f"sed -i 's/\"server_ip\": \"\"/\"server_ip\": \"{server_ip}\"/g' {rank_table}")
    if ret != 0:
        print(f"Failed to set server_ip in {rank_table}")
        return False
    for i in range(rank_size):
        ret = os.system(f"sed -i 's/\"device_ip\": \"\", \"rank_id\": \"{i}\"/"
                        f"\"device_ip\": \"{device_ips[i]}\", \"rank_id\": \"{i}\"/' {rank_table}")
        if ret != 0:
            print(f"Failed to set device_ip for rank {i} in {rank_table}")
            return False
    return True


def msrun_cross_cluster(num_cluster=2, rank_size=8):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # set server_ip and device_ips in rank table json file
    origin_rank_table = f"cross_{num_cluster}_cluster_rank_table.json"
    rank_table = "cross_cluster_rank_table.json"
    os.system(f"cp {origin_rank_table} {rank_table}")
    set_success = set_cross_cluster_rank_table(rank_size=rank_size, rank_table=rank_table)
    if not set_success:
        return False, "set rank table file failed, please check the ms_run log and rank table file"

    # run test
    result = subprocess.getoutput(
        f"msrun --worker_num=8 --local_worker_num=8 --rank_table_file={rank_table} --master_addr=127.0.0.1 "\
        f"--master_port=8118 --log_dir=ms_run --join=True --cluster_time_out=300 cross_cluster_net.py"
    )
    test_passed = "test all cross cluster communication operators success" in result and \
                  "test some cross cluster communication operators failed, please check the log" not in result

    if not test_passed:
        return False, "test some cross cluster communication operators failed, please check the ms_run log"
    return True, f"test cross cluster {num_cluster} az passed"


def test_cross_cluster_2_az():
    """
    Feature: test cross cluster between 2 az
    Description: Test CCOOL communication library in cross 2az scenarios.
    Expectation: Run success, all CCOOL communication operators in all workers pass the test.
    """
    result, msg = msrun_cross_cluster(num_cluster=2, rank_size=8)
    assert result, msg


def test_cross_cluster_4_az():
    """
    Feature: test cross cluster in 4 az
    Description: Test CCOOL communication library in cross 4az scenarios.
    Expectation: Run success, all CCOOL communication operators in all workers pass the test.
    """
    result, msg = msrun_cross_cluster(num_cluster=4, rank_size=8)
    assert result, msg
