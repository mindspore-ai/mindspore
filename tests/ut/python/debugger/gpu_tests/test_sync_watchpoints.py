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

import shutil
import numpy as np
import mindspore.offline_debug.dbg_services as d
from dump_test_utils import compare_actual_with_expected, build_dump_structure

GENERATE_GOLDEN = False
test_name = "sync_watchpoints"


def test_sync_trans_false_watchpoints():

    if GENERATE_GOLDEN:
        f_write = open(test_name + ".expected", "w")
    else:
        f_write = open(test_name + ".actual", "w")

    name1 = "Conv2D.Conv2D-op369.0.0."
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

    name2 = "Parameter.fc2.bias.0.0."
    tensor2 = np.array([-5.0167350e-06, 1.2509107e-05, -4.3148934e-06, 8.1415592e-06,
                        2.1177532e-07, 2.9952851e-06], np.float32)
    info2 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/Parameter[6]_11/fc2.bias",
                         slot=0, iteration=2, rank_id=0, root_graph_id=0, is_output=True)

    tensor3 = np.array([2.9060817e-07, -5.1009415e-06, -2.8662325e-06, 2.6036503e-06,
                        -5.1546101e-07, 6.0798648e-06], np.float32)
    info3 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/Parameter[6]_11/fc2.bias",
                         slot=0, iteration=3, rank_id=0, root_graph_id=0, is_output=True)

    name3 = "Parameter.fc3.bias.0.0."
    tensor4 = np.array([2.2930422e-04, -3.6369250e-04, 7.1337068e-04, -1.9567949e-05], np.float32)
    info4 = d.TensorInfo(node_name="Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/Parameter[6]_11/fc3.bias",
                         slot=0, iteration=2, rank_id=0, root_graph_id=0, is_output=True)

    tensor_info = [info1, info2, info3, info4]
    tensor_name = [name1, name2, name2, name3]
    tensor_list = [tensor1, tensor2, tensor3, tensor4]

    temp_dir = build_dump_structure(tensor_name, tensor_list, "alexnet", tensor_info)

    debugger_backend = d.DbgServices(dump_file_path=temp_dir)

    _ = debugger_backend.initialize(net_name="alexnet", is_sync_mode=True)
    # NOTES:
    # -> watch_condition=6 is MIN_LT
    # -> watch_condition=18 is CHANGE_TOO_LARGE
    # -> watch_condition=20 is NOT_CHANGE

    # test 1: watchpoint set and hit (watch_condition=6)
    param1 = d.Parameter(name="param", disabled=False, value=0.0)
    _ = debugger_backend.add_watchpoint(watchpoint_id=1, watch_condition=6,
                                        check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/conv1-Conv2d/"
                                                         "Conv2D-op369":
                                                         {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                          }}, parameter_list=[param1])

    watchpoint_hits_test_1 = debugger_backend.check_watchpoints(iteration=2)
    if len(watchpoint_hits_test_1) != 1:
        f_write.write(
            "ERROR -> test 1: watchpoint set but not hit just once\n")
    print_watchpoint_hits(watchpoint_hits_test_1, 1, f_write)

    # test 2: watchpoint remove and ensure it's not hit
    _ = debugger_backend.remove_watchpoint(watchpoint_id=1)
    watchpoint_hits_test_2 = debugger_backend.check_watchpoints(iteration=2)
    if watchpoint_hits_test_2:
        f_write.write("ERROR -> test 2: watchpoint removed but hit\n")

    # test 3: watchpoint set and not hit, then remove
    param2 = d.Parameter(name="param", disabled=False, value=-1000.0)
    _ = debugger_backend.add_watchpoint(watchpoint_id=2, watch_condition=6,
                                        check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/conv1-Conv2d/"
                                                         "Conv2D-op369":
                                                         {"rank_id": [0], "root_graph_id": [0], "is_output": False
                                                          }}, parameter_list=[param2])

    watchpoint_hits_test_3 = debugger_backend.check_watchpoints(iteration=2)
    if watchpoint_hits_test_3:
        f_write.write(
            "ERROR -> test 3: watchpoint set but not supposed to be hit\n")
    _ = debugger_backend.remove_watchpoint(watchpoint_id=2)

    # test 4: weight change watchpoint set and hit
    param_abs_mean_update_ratio_gt = d.Parameter(
        name="abs_mean_update_ratio_gt", disabled=False, value=0.0)
    param_epsilon = d.Parameter(name="epsilon", disabled=True, value=0.0)
    _ = debugger_backend.add_watchpoint(watchpoint_id=3, watch_condition=18,
                                        check_node_list={"Default/network-WithLossCell/_backbone-AlexNet/fc3-Dense/"
                                                         "Parameter[6]_11/fc2.bias":
                                                         {"rank_id": [0], "root_graph_id": [0], "is_output": True
                                                          }}, parameter_list=[param_abs_mean_update_ratio_gt,
                                                                              param_epsilon])

    watchpoint_hits_test_4 = debugger_backend.check_watchpoints(iteration=3)
    if len(watchpoint_hits_test_4) != 1:
        f_write.write("ERROR -> test 4: watchpoint weight change set but not hit just once\n")
    print_watchpoint_hits(watchpoint_hits_test_4, 4, f_write)
    f_write.close()
    shutil.rmtree(temp_dir)
    if not GENERATE_GOLDEN:
        assert compare_actual_with_expected(test_name)


def print_watchpoint_hits(watchpoint_hits, test_id, f_write):
    """Print watchpoint hits."""
    for x, _ in enumerate(watchpoint_hits):
        f_write.write(
            "-----------------------------------------------------------\n")
        f_write.write("watchpoint_hit for test_%u attributes:" %
                      test_id + "\n")
        f_write.write("name =  " + watchpoint_hits[x].name + "\n")
        f_write.write("slot =  " + str(watchpoint_hits[x].slot) + "\n")
        f_write.write("condition =  " +
                      str(watchpoint_hits[x].condition) + "\n")
        f_write.write("watchpoint_id =  " +
                      str(watchpoint_hits[x].watchpoint_id) + "\n")
        for p, _ in enumerate(watchpoint_hits[x].parameters):
            f_write.write("parameter  " + str(p) + "  name =  " +
                          watchpoint_hits[x].parameters[p].name + "\n")
            f_write.write("parameter  " + str(p) + "  disabled =  " +
                          str(watchpoint_hits[x].parameters[p].disabled) + "\n")
            f_write.write("parameter  " + str(p) + "  value =  " +
                          str(watchpoint_hits[x].parameters[p].value) + "\n")
            f_write.write("parameter  " + str(p) + "  hit =  " +
                          str(watchpoint_hits[x].parameters[p].hit) + "\n")
            f_write.write("parameter  " + str(p) + "  actual_value =  " +
                          str(watchpoint_hits[x].parameters[p].actual_value) + "\n")
        f_write.write("error code =  " +
                      str(watchpoint_hits[x].error_code) + "\n")
        f_write.write("rank_id =  " +
                      str(watchpoint_hits[x].rank_id) + "\n")
        f_write.write("root_graph_id =  " +
                      str(watchpoint_hits[x].root_graph_id) + "\n")


if __name__ == "__main__":
    test_sync_trans_false_watchpoints()
