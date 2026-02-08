# Copyright 2024 Huawei Technologies Co., Ltd
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

"""test fmea cases"""
import os
import json
import subprocess
import mindspore as ms
from mindspore import context
from tests.st.auto_parallel.utils.fmea_utils import check_key_word_in_file, check_comm_order_same
from tests.mark_utils import arg_mark


def _create_rank_table_file(save_json_to_path, rank_table_dict):
    """
    create rank table file for train or test
    """
    with open(save_json_to_path, "w", encoding="utf-8") as f:
        json.dump(rank_table_dict, f)


def _get_device_ips():
    device_ips = []
    result = subprocess.getoutput("for i in {0..7};do hccn_tool -i $i -ip -g;done")
    lines = result.splitlines()
    for line in lines:
        key, value = line.split(':')
        if key == 'ipaddr':
            device_ips.append(value)
    return device_ips


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_repeated_rank_id():
    '''
    Feature: Auto Parallel fault test case.
    Description: Test different workers were assigned repeated rank id.
    Expectation: Run success
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # run parallel network
    ret = os.system("""
    export MS_TOPO_TIMEOUT=10
    cp ./scripts/auto_parallel_fault_error_rank_id.sh ./scripts/auto_parallel_fault_error_rank_id_temp.sh
    sed -i 's/export MS_NODE_ID=$i/export MS_NODE_ID=6/' ./scripts/auto_parallel_fault_error_rank_id_temp.sh
    bash ./scripts/auto_parallel_fault_error_rank_id_temp.sh
    """)
    assert ret == 0

    # check error msg
    error_log = "error_msg.log"
    error_msg = "Failed to register the compute graph node: 6. Reason: Repeated registration node: 6 to the scheduler."
    ret = os.system("grep -r RuntimeError ./device &> error_msg.log")
    assert ret == 0
    assert check_key_word_in_file(error_msg, error_log)

    # clean workspace
    os.system("""
    rm -rf device
    rm -rf ./scripts/auto_parallel_fault_error_rank_id_temp.sh
    rm -rf error_msg.log
    """)


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_repeated_rank_id_ms_run():
    '''
    Feature: Auto Parallel fault test case.
    Description: Test different workers were assigned repeated rank id.
    Expectation: Run success
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # get a list of real device ips in context.
    device_ips = _get_device_ips()
    # create rank table file for 2 devices with repeated rank ids.
    rank_table_dict_2p = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [{
            "server_id": "10.*.*.*",
            "device": [{"device_id": "0", "device_ip": "192.1.*.6", "rank_id": "1"},
                       {"device_id": "1", "device_ip": "192.2.*.6", "rank_id": "1"}],
            "host_nic_ip": "reserve",
            "pod_ip": "127.0.0.1"
            }],
        "status": "completed"
        }
    rank_table_dict_2p["server_list"][0]["device"][0]["device_ip"] = device_ips[0]
    rank_table_dict_2p["server_list"][0]["device"][1]["device_ip"] = device_ips[1]
    _create_rank_table_file("rank_table_2p.json", rank_table_dict_2p)

    # run test
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    result = subprocess.getoutput(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=8118 "\
        "--join=True --rank_table_file=rank_table_2p.json --log_dir=./repeated_rank_id "\
        "auto_parallel_fault_error_rank_id_net.py"
    )
    # The current repeatability check of ms_run is performed before importing the rank table,
    # so the current ms_run error message needs to be used as the check information
    assert result.find("RuntimeError: Failed to AllGather host's hash name due to timeout.") != -1

    # clean workspace
    os.system("rm -rf repeated_rank_id rank_table_2p.json error_msg.log")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_out_of_range_rank_id():
    '''
    Feature: Auto Parallel fault test case.
    Description: Test workers were assigned out of range rank id.
    Expectation: Run success
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    # get a list of real device ips in context.
    device_ips = _get_device_ips()
    # create rank table file for 2 devices with repeated rank ids.
    rank_table_dict_2p = {
        "version": "1.0",
        "server_count": "1",
        "server_list": [{
            "server_id": "10.*.*.*",
            "device": [{"device_id": "0", "device_ip": "192.1.*.6", "rank_id": "3"},
                       {"device_id": "1", "device_ip": "192.2.*.6", "rank_id": "3"}],
            "host_nic_ip": "reserve",
            "pod_ip": "127.0.0.1"
            }],
        "status": "completed"
        }
    rank_table_dict_2p["server_list"][0]["device"][0]["device_ip"] = device_ips[0]
    rank_table_dict_2p["server_list"][0]["device"][1]["device_ip"] = device_ips[1]
    _create_rank_table_file("rank_table_2p.json", rank_table_dict_2p)

    # run test
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
    result = subprocess.getoutput(
        "msrun --worker_num=2 --local_worker_num=2 --master_addr=127.0.0.1 --master_port=8118 "\
        "--join=True --rank_table_file=rank_table_2p.json --log_dir=./out_of_range_rank_id "\
        "auto_parallel_fault_error_rank_id_net.py"
    )
    assert result.find("The global rank id 3 should be less than global rank size 2") != -1

    # clean workspace
    os.system("rm -rf out_of_range_rank_id rank_table_2p.json error_msg.log")


@arg_mark(plat_marks=["platform_ascend910b"], level_mark="level1", card_mark="allcards", essential_mark="essential")
def test_error_comm_order():
    '''
    Feature: Auto Parallel fault test case.
    Description: Test different workers have inconsistent execution order of communication operators.
    Expectation: Run success
    '''
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # run test
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_level='O0')
    os.environ['GLOG_v'] = str(1)
    subprocess.run(
        "msrun --worker_num=8 --local_worker_num=8 --master_addr=127.0.0.1 --master_port=8118 --join=True "
        "--log_dir=./error_comm_order pytest -s --disable-warnings "
        "auto_parallel_fault_error_comm_order_net.py::test_auto_parallel_fault_error_comm_order",
        shell=True,
        check=False,
        capture_output=False
    )

    # check error msg
    assert not check_comm_order_same(rank_size=8, log_path="./error_comm_order")
    # clean workspace
    os.system("rm -rf error_comm_order error_msg.log")
