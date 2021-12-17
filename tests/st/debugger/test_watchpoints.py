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
# ==============================================================================
"""
Watchpoints test script for offline debugger APIs.
"""

import os
import json
import time
import tempfile
import numpy as np
import pytest
import mindspore.offline_debug.dbg_services as d
from tests.security_utils import security_off_wrap
from dump_test_utils import build_dump_structure, write_watchpoint_to_json

GENERATE_GOLDEN = False
watchpoint_hits_json = []


def run_watchpoints(is_sync):
    if is_sync:
        test_name = "sync_watchpoints"
    else:
        test_name = "async_watchpoints"

    name1 = "Conv2D.Conv2D-op369.0.0.1"
    tensor1 = np.array([[[-1.2808e-03, 7.7629e-03, 1.9241e-02],
                         [-1.3931e-02, 8.9359e-04, -1.1520e-02],
                         [-6.3248e-03, 1.8749e-03, 1.0132e-02]],
                        [[-2.5520e-03, -6.0005e-03, -5.1918e-03],
                         [-2.7866e-03, 2.5487e-04, 8.4782e-04],
                         [-4.6310e-03, -8.9111e-03, -8.1778e-05]],
                        [[1.3914e-03, 6.0844e-04, 1.0643e-03],
                         [-2.0966e-02, -1.2865e-03, -1.8692e-03],
                         [-1.6647e-02, 1.0233e-03, -4.1313e-03]]], np.float32)
    info1 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/conv1-Conv2d/Conv2D-op369",
                         slot=1, iteration=2, rank_id=0, root_graph_id=0, is_output=False)

    name2 = "Parameter.fc2.bias.0.0.2"
    tensor2 = np.array([-5.0167350e-06, 1.2509107e-05, -4.3148934e-06, 8.1415592e-06,
                        2.1177532e-07, 2.9952851e-06], np.float32)
    info2 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                                   "Parameter[6]_11/fc2.bias",
                         slot=0, iteration=2, rank_id=0, root_graph_id=0, is_output=True)

    tensor3 = np.array([2.9060817e-07, -5.1009415e-06, -2.8662325e-06, 2.6036503e-06,
                        -5.1546101e-07, 6.0798648e-06], np.float32)
    info3 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                                   "Parameter[6]_11/fc2.bias",
                         slot=0, iteration=3, rank_id=0, root_graph_id=0, is_output=True)

    tensor_info = [info1, info2, info3]
    tensor_name = [name1, name2, name2]
    tensor_list = [tensor1, tensor2, tensor3]

    pwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir=pwd) as tmp_dir:
        temp_dir = build_dump_structure(tmp_dir, tensor_name, tensor_list, "Test", tensor_info)

        debugger_backend = d.DbgServices(dump_file_path=temp_dir)
        debugger_backend.initialize(net_name="Test", is_sync_mode=is_sync)

        # NOTES:
        # -> watch_condition=6 is MIN_LT
        # -> watch_condition=18 is CHANGE_TOO_LARGE

        # test 1: watchpoint set and hit (watch_condition=6)
        param1 = d.Parameter(name="param", disabled=False, value=0.0)
        debugger_backend.add_watchpoint(watchpoint_id=1, watch_condition=6,
                                        check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/"
                                                         "conv1-Conv2d/Conv2D-op369":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param1])

        watchpoint_hits_test_1 = debugger_backend.check_watchpoints(iteration=2)
        assert len(watchpoint_hits_test_1) == 1
        if GENERATE_GOLDEN:
            print_watchpoint_hits(watchpoint_hits_test_1, 0, False, test_name)
        else:
            compare_expect_actual_result(watchpoint_hits_test_1, 0, test_name)

        # test 2: watchpoint remove and ensure it's not hit
        debugger_backend.remove_watchpoint(watchpoint_id=1)
        watchpoint_hits_test_2 = debugger_backend.check_watchpoints(iteration=2)
        assert not watchpoint_hits_test_2

        # test 3: watchpoint set and not hit, then remove
        param2 = d.Parameter(name="param", disabled=False, value=-1000.0)
        debugger_backend.add_watchpoint(watchpoint_id=2, watch_condition=6,
                                        check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/"
                                                         "conv1-Conv2d/Conv2D-op369":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                              }}, parameter_list=[param2])

        watchpoint_hits_test_3 = debugger_backend.check_watchpoints(iteration=2)
        assert not watchpoint_hits_test_3
        _ = debugger_backend.remove_watchpoint(watchpoint_id=2)

        # test 4: weight change watchpoint set and hit
        param_abs_mean_update_ratio_gt = d.Parameter(
            name="abs_mean_update_ratio_gt", disabled=False, value=0.0)
        param_epsilon = d.Parameter(name="epsilon", disabled=True, value=0.0)
        debugger_backend.add_watchpoint(watchpoint_id=3, watch_condition=18,
                                        check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                                                         "Parameter[6]_11/fc2.bias":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": True
                                                              }}, parameter_list=[param_abs_mean_update_ratio_gt,
                                                                                  param_epsilon])

        watchpoint_hits_test_4 = debugger_backend.check_watchpoints(iteration=3)
        assert len(watchpoint_hits_test_4) == 1

        if GENERATE_GOLDEN:
            print_watchpoint_hits(watchpoint_hits_test_4, 1, True, test_name)
        else:
            compare_expect_actual_result(watchpoint_hits_test_4, 1, test_name)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_sync_watchpoints():
    run_watchpoints(True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_watchpoints():
    run_watchpoints(False)


def run_overflow_watchpoint(is_overflow):
    test_name = "overflow_watchpoint"
    tensor = np.array([65504, 65504], np.float16)
    task_id = 2
    stream_id = 7
    pwd = os.getcwd()
    with tempfile.TemporaryDirectory(dir=pwd) as tmp_dir:
        path = os.path.join(tmp_dir, "rank_0", "Add", "0", "0")
        os.makedirs(path, exist_ok=True)
        add_file = os.path.join(path, "Add.Default_Add-op0." + str(task_id) + "." + str(stream_id) + "."
                                + str(int(round(time.time() * 1000000))))
        with open(add_file, 'wb') as add_f:
            add_f.write(b'1')
            add_f.seek(8)
            add_f.write(b'\n\x032.0\x10\x83\xf7\xef\x9f\x99\xc8\xf3\x02\x1a\x10\x08\x02\x10\x02\x1a\x03')
            add_f.write(b'\n\x01\x020\x04:\x03\n\x01\x022\x0f')
            add_f.write(b'Default/Add-op0')
            add_f.write(tensor)
        overflow_file = os.path.join(path, "Opdebug.Node_OpDebug." + str(task_id) + "." + str(stream_id) +
                                     "." + str(int(round(time.time() * 1000000))))
        with open(overflow_file, 'wb') as f:
            f.seek(321, 0)
            byte_list = []
            for i in range(256):
                if i == 16:
                    byte_list.append(stream_id)
                elif i == 24:
                    if is_overflow:
                        byte_list.append(task_id)
                    else:
                        # wrong task_id, should not generate overflow watchpoint hit
                        byte_list.append(task_id + 1)
                else:
                    byte_list.append(0)
            new_byte_array = bytearray(byte_list)
            f.write(bytes(new_byte_array))
        debugger_backend = d.DbgServices(dump_file_path=tmp_dir)
        debugger_backend.initialize(net_name="Add", is_sync_mode=False)
        debugger_backend.add_watchpoint(watchpoint_id=1, watch_condition=2,
                                        check_node_list={"Default/Add-op0":
                                                             {"rank_id": [0], "root_graph_id": [0], "is_output": True
                                                              }}, parameter_list=[])

        watchpoint_hits_test = debugger_backend.check_watchpoints(iteration=0)

        if is_overflow:
            assert len(watchpoint_hits_test) == 1
            if GENERATE_GOLDEN:
                print_watchpoint_hits(watchpoint_hits_test, 0, True, test_name)
            else:
                compare_expect_actual_result(watchpoint_hits_test, 0, test_name)
        else:
            assert not watchpoint_hits_test


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@security_off_wrap
def test_async_overflow_watchpoints_hit():
    """
    Feature: Offline Debugger CheckWatchpoint
    Description: Test check overflow watchpoint hit
    Expectation: Overflow watchpoint is hit
    """
    run_overflow_watchpoint(True)


def compare_expect_actual_result(watchpoint_hits_list, test_index, test_name):
    """Compare actual result with golden file."""
    pwd = os.getcwd()
    golden_file = os.path.realpath(os.path.join(pwd, "golden", test_name + "_expected.json"))
    with open(golden_file) as f:
        expected_list = json.load(f)
        for x, watchpoint_hits in enumerate(watchpoint_hits_list):
            test_id = "watchpoint_hit" + str(test_index + x + 1)
            expect_wp = expected_list[x + test_index][test_id]
            actual_wp = write_watchpoint_to_json(watchpoint_hits)
            assert actual_wp == expect_wp

def print_watchpoint_hits(watchpoint_hits_list, test_index, is_print, test_name):
    """Print watchpoint hits."""
    for x, watchpoint_hits in enumerate(watchpoint_hits_list):
        watchpoint_hit = "watchpoint_hit" + str(test_index + x + 1)
        wp = write_watchpoint_to_json(watchpoint_hits)
        watchpoint_hits_json.append({watchpoint_hit: wp})
    if is_print:
        with open(test_name + "_expected.json", "w") as dump_f:
            json.dump(watchpoint_hits_json, dump_f, indent=4, separators=(',', ': '))
